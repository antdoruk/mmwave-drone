"""Parser for the UART output of the TI mmWave SDK 3.x out-of-box demo (xwr68xx).

Packet layout (little endian):
    magic word   8 bytes   02 01 04 03 06 05 08 07
    version      u32
    total length u32   (whole packet, header included, padded to 32 bytes)
    platform     u32
    frame number u32
    cpu cycles   u32
    num objects  u32
    num TLVs     u32
    subframe     u32       -> header is 40 bytes
then num TLVs blocks of:  type u32, length u32 (payload only), payload.

The parser is a stream parser: feed() it whatever bytes the serial port gives,
and it returns every complete frame it finds. It never trusts a length field
without checking it against what is actually in the buffer.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = bytes([2, 1, 4, 3, 6, 5, 8, 7])
HEADER = struct.Struct("<8sIIIIIIII")          # 40 bytes
TLV_HDR = struct.Struct("<II")

TLV_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_NOISE_PROFILE = 3
TLV_AZ_HEATMAP = 4
TLV_RD_HEATMAP = 5
TLV_STATS = 6
TLV_SIDE_INFO = 7
TLV_AZEL_HEATMAP = 8
TLV_TEMPERATURE = 9

# Log-magnitude TLVs are log2(|x|) in Q9 fixed point (SDK docs). Converting to dB:
# 20*log10(2) dB per unit of log2, divided by 2**9.
Q_LOG = 9
LOG2_TO_DB = 20 * np.log10(2) / (1 << Q_LOG)

MAX_PACKET = 1 << 20


@dataclass
class Frame:
    number: int
    subframe: int
    num_obj: int
    timestamp: float = 0.0                      # host time when the frame was completed
    points: np.ndarray | None = None            # (N, 4): x, y, z, v
    snr_db: np.ndarray | None = None            # (N,)
    noise_db: np.ndarray | None = None          # (N,)
    range_profile: np.ndarray | None = None     # dB, (range_bins,)
    noise_profile: np.ndarray | None = None     # dB, (range_bins,)
    rd_raw: np.ndarray | None = None            # raw uint16, flat
    stats: dict = field(default_factory=dict)
    unknown_tlvs: list = field(default_factory=list)

    def rd_db(self, range_bins: int, doppler_bins: int) -> np.ndarray | None:
        """Range-Doppler map in dB, shape (range, doppler), zero velocity in the middle."""
        if self.rd_raw is None or self.rd_raw.size != range_bins * doppler_bins:
            return None
        m = self.rd_raw.astype(np.float32).reshape(range_bins, doppler_bins) * LOG2_TO_DB
        return np.fft.fftshift(m, axes=1)


class StreamParser:
    def __init__(self):
        self.buf = bytearray()
        self.frames_ok = 0
        self.bytes_dropped = 0
        self.bad_packets = 0

    def feed(self, data: bytes, timestamp: float = 0.0) -> list[Frame]:
        self.buf.extend(data)
        out = []
        while True:
            i = self.buf.find(MAGIC)
            if i < 0:
                # keep the last 7 bytes, a magic word may be split between reads
                keep = min(len(self.buf), len(MAGIC) - 1)
                self.bytes_dropped += len(self.buf) - keep
                del self.buf[: len(self.buf) - keep]
                return out
            if i > 0:
                self.bytes_dropped += i
                del self.buf[:i]
            if len(self.buf) < HEADER.size:
                return out
            (_, version, total_len, platform, frame_no, cycles,
             num_obj, num_tlv, subframe) = HEADER.unpack_from(self.buf, 0)
            if total_len < HEADER.size or total_len > MAX_PACKET or num_tlv > 32:
                # corrupt header: skip this magic word and resync
                self.bad_packets += 1
                del self.buf[:1]
                continue
            if len(self.buf) < total_len:
                return out
            packet = bytes(self.buf[:total_len])
            nxt = self.buf.find(MAGIC, 1)
            if 0 < nxt < total_len:
                # another packet starts inside this one: the first was truncated
                self.bad_packets += 1
                del self.buf[:nxt]
                continue
            del self.buf[:total_len]
            try:
                fr = parse_packet(packet, timestamp)
            except (ValueError, struct.error):   # malformed payload: drop it, keep streaming
                fr = None
            if fr is None:
                self.bad_packets += 1
                continue
            self.frames_ok += 1
            out.append(fr)


def parse_packet(packet: bytes, timestamp: float = 0.0) -> Frame | None:
    (_, version, total_len, platform, frame_no, cycles,
     num_obj, num_tlv, subframe) = HEADER.unpack_from(packet, 0)
    fr = Frame(number=frame_no, subframe=subframe, num_obj=num_obj, timestamp=timestamp)
    off = HEADER.size
    for _ in range(num_tlv):
        if off + TLV_HDR.size > len(packet):
            return None
        t, ln = TLV_HDR.unpack_from(packet, off)
        off += TLV_HDR.size
        if off + ln > len(packet):
            return None
        payload = packet[off: off + ln]
        off += ln
        # 16-bit payloads must have an even length, side info 4 bytes per detected point
        if t in (TLV_RANGE_PROFILE, TLV_NOISE_PROFILE, TLV_RD_HEATMAP) and ln % 2:
            return None
        if t == TLV_SIDE_INFO and ln != 4 * num_obj:
            return None
        if t == TLV_POINTS:
            if ln != 16 * num_obj:
                return None
            fr.points = np.frombuffer(payload, dtype="<f4").reshape(-1, 4).copy()
        elif t == TLV_SIDE_INFO:
            si = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
            fr.snr_db = si[:, 0] * 0.1
            fr.noise_db = si[:, 1] * 0.1
        elif t == TLV_RANGE_PROFILE:
            fr.range_profile = np.frombuffer(payload, dtype="<u2") * LOG2_TO_DB
        elif t == TLV_NOISE_PROFILE:
            fr.noise_profile = np.frombuffer(payload, dtype="<u2") * LOG2_TO_DB
        elif t == TLV_RD_HEATMAP:
            fr.rd_raw = np.frombuffer(payload, dtype="<u2").copy()
        elif t == TLV_STATS and ln >= 24:
            keys = ("interframe_proc_us", "transmit_out_us", "interframe_margin_us",
                    "interchirp_margin_us", "active_cpu_load", "interframe_cpu_load")
            fr.stats = dict(zip(keys, struct.unpack_from("<6I", payload, 0)))
        else:
            fr.unknown_tlvs.append((t, ln))
    return fr


# ---------------------------------------------------------------------------
# Encoder: builds packets exactly like the firmware does. Used by the simulator
# (so the whole pipeline can run without the board) and by the tests.
# ---------------------------------------------------------------------------
def encode_packet(frame_no: int, points=None, range_profile_db=None, noise_profile_db=None,
                  rd_db_unshifted=None, snr_db=None, stats=None) -> bytes:
    tlvs = []
    n_obj = 0
    if points is not None and len(points):
        pts = np.asarray(points, dtype="<f4").reshape(-1, 4)
        n_obj = len(pts)
        tlvs.append((TLV_POINTS, pts.tobytes()))
    if range_profile_db is not None:
        tlvs.append((TLV_RANGE_PROFILE, _db_to_q(range_profile_db).tobytes()))
    if noise_profile_db is not None:
        tlvs.append((TLV_NOISE_PROFILE, _db_to_q(noise_profile_db).tobytes()))
    if rd_db_unshifted is not None:
        tlvs.append((TLV_RD_HEATMAP, _db_to_q(rd_db_unshifted).ravel().tobytes()))
    if n_obj and snr_db is not None:
        si = np.zeros((n_obj, 2), dtype="<i2")
        si[:, 0] = np.round(np.asarray(snr_db) * 10)
        tlvs.append((TLV_SIDE_INFO, si.tobytes()))
    if stats is not None:
        tlvs.append((TLV_STATS, struct.pack("<6I", *stats)))
    body = b"".join(TLV_HDR.pack(t, len(p)) + p for t, p in tlvs)
    total = HEADER.size + len(body)
    pad = (-total) % 32
    total += pad
    hdr = HEADER.pack(MAGIC, 0x03050004, total, 0xA6843, frame_no, 0, n_obj, len(tlvs), 0)
    return hdr + body + bytes(pad)


def _db_to_q(db) -> np.ndarray:
    q = np.round(np.asarray(db, dtype=np.float64) / LOG2_TO_DB)
    return np.clip(q, 0, 65535).astype("<u2")
