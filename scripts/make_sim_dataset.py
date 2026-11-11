"""Generate simulated recordings in exactly the format of real ones.

Used to (1) develop and test the whole pipeline without hardware and
(2) compare simulation with measurement (report/make_figures.py, sim_vs_meas).

    python3 scripts/make_sim_dataset.py --out data_sim --seconds 15
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import numpy as np

from mmwd.config import parse_cfg
from mmwd.session import save_session
from mmwd.sim import Drone, Fan, Human, Simulator, Static, detections_from_rd
from mmwd.tlv import StreamParser, encode_packet

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="configs/md_1tx.cfg")
ap.add_argument("--out", default="data_sim")
ap.add_argument("--seconds", type=float, default=15)
ap.add_argument("--reps", type=int, default=2, help="recordings per class and distance")
args = ap.parse_args()

p = parse_cfg(args.cfg)
Path(args.out).mkdir(exist_ok=True)
rng = np.random.default_rng(0)
room = [Static((0.4, 4.5, 0.0), 0.5), Static((-1.0, 6.0, 0.3), 1.0)]


def make(label, d, k):
    s = int(rng.integers(1e9))
    if label == "drone":
        return [Drone(hub=(rng.uniform(-0.2, 0.2), d, 0), rps=rng.uniform(350, 600), seed=s),
                Static((0, d + 0.6, 0), 0.01)] + room
    if label == "human":
        return [Human(start=(rng.uniform(-0.3, 0.3), d, 0), walk_speed=rng.uniform(0.6, 1.5),
                      heading_deg=rng.uniform(0, 60), seed=s)] + room
    if label == "fan":
        return [Fan(center=(0, d, 0), rps=rng.uniform(12, 25), blades=int(rng.choice([3, 5])))] + room
    return room


n = int(args.seconds * p.fps)
for label in ("empty", "drone", "human", "fan"):
    for d in (1.0, 2.0, 3.0, 5.0):
        for k in range(args.reps):
            sim = Simulator(p, seed=int(rng.integers(1e9)))
            tg = make(label, d, k)
            frames, parser = [], StreamParser()
            for i, (t0, rd, rp) in enumerate(sim.frames(tg, n, t_start=rng.uniform(0, 10))):
                pkt = encode_packet(i, points=detections_from_rd(rd, p), range_profile_db=rp,
                                    rd_db_unshifted=rd)
                for f in parser.feed(pkt, t0):
                    frames.append(f)
            stem = Path(args.out) / f"sim_{label}_{d:g}m_{k}"
            save_session(stem, frames, p, Path(args.cfg).read_text(),
                         dict(label=label, distance_m=d, angle_deg=0.0, source="sim", note=""))
            print("wrote", stem.with_suffix(".npz"), len(frames), "frames")
