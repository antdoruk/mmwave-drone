"""Talking to the IWR6843AOPEVM over USB, and a fake board with the same interface.

The EVM shows up as two serial ports:
  * CLI port  (115200 baud): we type the .cfg lines here, the board answers "Done".
  * Data port (921600 baud): the board streams binary TLV packets here.
On a Mac they look like /dev/cu.SLAB_USBtoUART + /dev/cu.SLAB_USBtoUART3
(Silicon Labs driver) or /dev/cu.usbserial-XXXX0 + ...1 (Apple's driver).
We do not guess which is which: we send "version" to each and see who answers.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

from .config import RadarParams, read_cfg_lines
from .tlv import StreamParser

CLI_BAUD = 115200
DATA_BAUD = 921600
PORT_HINTS = ("SLAB", "usbserial", "usbmodem", "XDS", "CP210", "ttyUSB", "ttyACM")


def candidate_ports() -> list[str]:
    from serial.tools import list_ports
    ports = []
    for p in list_ports.comports():
        name = p.device
        desc = f"{p.description} {p.manufacturer or ''} {p.hwid}"
        if any(h.lower() in (name + desc).lower() for h in PORT_HINTS):
            if name.startswith("/dev/tty."):     # on macOS use the cu.* twin
                continue
            ports.append(name)
    return sorted(ports)


def _ask(ser, line: str, timeout: float = 1.0) -> str:
    ser.reset_input_buffer()
    ser.write((line + "\n").encode())
    t_end = time.time() + timeout
    out = b""
    while time.time() < t_end:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            out += chunk
            if b"Done" in out or b"Error" in out or b"mmwDemo:/>" in out.split(b"\n")[-1]:
                # small grace period for the rest of the line
                time.sleep(0.02)
                out += ser.read(ser.in_waiting)
                break
    return out.decode(errors="replace")


def find_cli_and_data(verbose: bool = True) -> tuple[str, str]:
    import serial
    ports = candidate_ports()
    if verbose:
        print("serial ports that look like the radar:", ports or "none")
    if len(ports) < 2:
        raise RuntimeError(
            "Need two serial ports from the board. Check the cable/adapter, run "
            "`ls /dev/cu.*`, and pass --cli and --data by hand if needed.")
    cli = None
    for p in ports:
        try:
            with serial.Serial(p, CLI_BAUD, timeout=0.1) as s:
                ans = _ask(s, "version", 1.5)
        except Exception as e:                      # port busy, permission, ...
            if verbose:
                print(f"  {p}: cannot open ({e})")
            continue
        if verbose:
            print(f"  {p}: {'answers CLI' if ans.strip() else 'silent'}")
        if "mmWave" in ans or "Platform" in ans or "mmwDemo" in ans or "Done" in ans:
            cli = p
            if verbose:
                print("    " + ans.strip().replace("\n", "\n    "))
            break
    if cli is None:
        raise RuntimeError(
            "No port answered the 'version' command. Is the out-of-box demo flashed and "
            "the board in functional mode (not flashing mode)? Try pressing the reset button.")
    # the data port is the sibling with the closest name
    others = [p for p in ports if p != cli]
    data = min(others, key=lambda p: _name_distance(p, cli))
    return cli, data


def _name_distance(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        n += x != y
    return n + abs(len(a) - len(b))


def send_cfg(cli_port: str, cfg_path: str | Path, verbose: bool = True) -> list[str]:
    """Send every line of the cfg. Returns the list of lines the board rejected."""
    import serial
    errors = []
    with serial.Serial(cli_port, CLI_BAUD, timeout=0.1) as s:
        for line in read_cfg_lines(cfg_path):
            ans = _ask(s, line, timeout=3.0 if line.startswith("sensorStart") else 1.0)
            ok = "Done" in ans
            if verbose:
                tag = "ok " if ok else "ERR"
                print(f"[{tag}] {line}")
                if not ok:
                    print("      board said: " + ans.strip().replace("\n", " | "))
            if not ok and not line.startswith("sensorStop"):
                errors.append(f"{line}  ->  {ans.strip()}")
    return errors


def stop_sensor(cli_port: str):
    import serial
    try:
        with serial.Serial(cli_port, CLI_BAUD, timeout=0.1) as s:
            _ask(s, "sensorStop", 1.0)
    except (serial.SerialException, OSError):
        pass                      # board already gone: nothing to stop


class FrameSource:
    """Common interface: start(), get(timeout) -> Frame | None, stop()."""
    raw_sink = None          # optional file object receiving raw bytes

    def start(self): ...
    def stop(self): ...
    def get(self, timeout: float = 1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class BoardSource(FrameSource):
    def __init__(self, data_port: str, raw_path: str | Path | None = None):
        self.data_port = data_port
        self.q = queue.Queue(maxsize=200)
        self.parser = StreamParser()
        self._stop = threading.Event()
        self.raw_path = raw_path
        self.thread = None
        self.lost = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        import serial
        raw = open(self.raw_path, "ab") if self.raw_path else None
        try:
            with serial.Serial(self.data_port, DATA_BAUD, timeout=0.05) as s:
                s.reset_input_buffer()
                while not self._stop.is_set():
                    chunk = s.read(s.in_waiting or 1)
                    if not chunk:
                        continue
                    if raw:
                        raw.write(chunk)
                    for fr in self.parser.feed(chunk, time.time()):
                        if self.q.full():
                            self.q.get_nowait()
                        self.q.put(fr)
        except (serial.SerialException, OSError) as e:
            self.lost = True
            print(f"\nCONNECTION LOST: {e}\nThe USB link to the board dropped (loose cable?). "
                  "Reconnect it, check `ls /dev/cu.*` and run the command again.")
        finally:
            if raw:
                raw.close()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1)


class SimSource(FrameSource):
    """Runs the physics simulator in real time and pushes encoded packets through
    the same parser, so the rest of the code cannot tell the difference."""

    def __init__(self, params: RadarParams, scene_name: str = "drone", distance: float = 2.0,
                 realtime: bool = True, seed: int = 0, raw_path=None):
        from .sim import Simulator, scene
        self.params = params
        self.sim = Simulator(params, seed=seed)
        self.targets = scene(scene_name, distance, seed)
        self.q = queue.Queue(maxsize=200)
        self.parser = StreamParser()
        self.realtime = realtime
        self._stop = threading.Event()
        self.raw_path = raw_path
        self.thread = None

    def set_scene(self, scene_name: str, distance: float = 2.0):
        from .sim import scene
        self.targets = scene(scene_name, distance)

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        from .sim import detections_from_rd
        from .tlv import encode_packet
        raw = open(self.raw_path, "ab") if self.raw_path else None
        k = 0
        period = self.params.frame_period_ms / 1000
        t_next = time.time()
        while not self._stop.is_set():
            t0 = k * period
            cube = self.sim.frame_cube(self.targets, t0)
            rd, rp = self.sim.process(cube)
            rd += 150
            rp += 150
            pts = detections_from_rd(rd, self.params)
            pkt = encode_packet(k, points=pts, range_profile_db=rp, rd_db_unshifted=rd,
                                snr_db=np.zeros(len(pts)) if len(pts) else None)
            if raw:
                raw.write(pkt)
            for fr in self.parser.feed(pkt, time.time()):
                if self.q.full():
                    self.q.get_nowait()
                self.q.put(fr)
            k += 1
            if self.realtime:
                t_next += period
                time.sleep(max(0.0, t_next - time.time()))
        if raw:
            raw.close()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1)


class ReplaySource(FrameSource):
    """Plays back a raw .bin capture at (roughly) the original frame rate."""

    def __init__(self, raw_path: str | Path, fps: float = 4.0, loop: bool = True):
        self.data = Path(raw_path).read_bytes()
        self.fps = fps
        self.loop = loop
        self.q = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            parser = StreamParser()
            frames = parser.feed(self.data, 0.0)
            for fr in frames:
                if self._stop.is_set():
                    return
                fr.timestamp = time.time()
                self.q.put(fr)
                time.sleep(1 / self.fps)
            if not self.loop:
                return

    def stop(self):
        self._stop.set()

