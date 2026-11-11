"""Record one labelled measurement.

    python3 scripts/record.py --label drone --distance 2.0 --angle 0 --seconds 20
    python3 scripts/record.py --label human --distance 3 --note "walking towards radar"
    python3 scripts/record.py --label empty --seconds 20
    python3 scripts/record.py --label hand_only --distance 3 --note "stick held, drone off"

Labels used by the classifier: empty, drone, human, fan. Anything else
(for example hand_only, drone_off) is kept for checks but not trained on.
Writes data/<prefix><nn>_<label>_<distance>m_<angle>deg.npz and the raw .bin next to it,
where <nn> is one more than the highest number already in the output folder.
"""
import argparse
import time
from pathlib import Path

import _path  # noqa: F401

from mmwd.cli import add_source_args, cfg_text, open_source
from mmwd.session import next_id, save_session

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_source_args(ap)
ap.add_argument("--label", required=True)
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--angle", type=float, default=0.0, help="drone/target yaw or azimuth angle, deg")
ap.add_argument("--note", default="")
ap.add_argument("--countdown", type=int, default=3)
ap.add_argument("--out", default="data")
ap.add_argument("--prefix", default="T", help="ID prefix, e.g. T for the test-day recordings")
args = ap.parse_args()

if args.label == "empty":
    args.distance = 0.0          # no target, distance is meaningless

Path(args.out).mkdir(exist_ok=True)
# no dots in the name: Path.with_suffix would treat "2.5m..." as an extension
dist = f"{args.distance:g}".replace(".", "p")
ang = f"{args.angle:g}".replace(".", "p")
stem = Path(args.out) / f"{next_id(args.out, args.prefix)}_{args.label}_{dist}m_{ang}deg"
if stem.with_suffix(".npz").exists() or stem.with_suffix(".bin").exists():
    raise SystemExit(f"{stem} already exists, not overwriting it")
src, p, cli = open_source(args, raw_path=stem.with_suffix(".bin"))

for i in range(args.countdown, 0, -1):
    print(f"starting in {i}...", flush=True)
    time.sleep(1)
src.start()
print(f"RECORDING {args.seconds:.0f} s  label={args.label}  distance={args.distance} m  angle={args.angle} deg")
frames = []
t0 = time.time()
while time.time() - t0 < args.seconds:
    f = src.get(timeout=1.0)
    if getattr(src, "lost", False):
        print("Recording stopped early: connection lost. Saving what arrived.")
        break
    if f is not None:
        frames.append(f)
        if len(frames) % max(1, int(p.fps)) == 0:
            print(f"  {time.time()-t0:5.1f} s  {len(frames)} frames", flush=True)
src.stop()
if cli:
    from mmwd.radar import stop_sensor
    stop_sensor(cli)

meta = dict(label=args.label, distance_m=args.distance, angle_deg=args.angle, note=args.note,
            seconds=args.seconds, source="sim:" + args.sim if args.sim else
            ("replay" if args.replay else "board"))
out = save_session(stem, frames, p, cfg_text(args.cfg), meta)
print(f"saved {out}  ({len(frames)} frames)")
