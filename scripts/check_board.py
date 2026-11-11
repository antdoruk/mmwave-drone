"""Board check: is the board alive, does it accept our chirp, do frames arrive intact?

    python3 scripts/check_board.py                 # real board
    python3 scripts/check_board.py --sim drone     # no board, simulator
"""
import argparse
import time

import _path  # noqa: F401
import numpy as np

from mmwd.cli import add_source_args, open_source

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_source_args(ap)
ap.add_argument("--seconds", type=float, default=5.0)
args = ap.parse_args()

src, p, cli = open_source(args)
print("\nWhat this chirp gives us:\n" + p.summary() + "\n")
src.start()
t0 = time.time()
frames = []
while time.time() - t0 < args.seconds:
    f = src.get(timeout=1.0)
    if f is not None:
        frames.append(f)
src.stop()

if not frames:
    print("No frames arrived. Typical causes: wrong data port, sensorStart failed, "
          "or the cable/adapter only carries power.")
    raise SystemExit(1)

dt = frames[-1].timestamp - frames[0].timestamp
nums = [f.number for f in frames]
gaps = int(np.sum(np.diff(nums) - 1)) if len(nums) > 1 else 0
rd_ok = sum(f.rd_raw is not None and f.rd_raw.size == p.range_bins * p.doppler_bins for f in frames)
fps = (len(frames) - 1) / dt if len(frames) > 1 and dt > 0 else float("nan")   # N frames span N-1 intervals
print(f"frames received      {len(frames)} in {dt:.1f} s  ->  {fps:.1f} fps "
      f"(expected {p.fps:.1f})")
print(f"missing frame numbers {gaps}")
print(f"frames with RD map   {rd_ok}/{len(frames)}")
pts = [len(f.points) if f.points is not None else 0 for f in frames]
print(f"points per frame     mean {np.mean(pts):.1f}, max {max(pts)}")
if hasattr(src, "parser"):
    print(f"bytes skipped        {src.parser.bytes_dropped}, bad packets {src.parser.bad_packets}")
if frames[-1].stats:
    print(f"board stats          {frames[-1].stats}")
ok = rd_ok == len(frames) and gaps <= 1
print("\nRESULT: " + ("all good, the board is ready (next: scripts/live.py)" if ok else "something is off, see the lines above"))
if not ok:
    raise SystemExit(1)
