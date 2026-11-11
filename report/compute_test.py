"""Numbers for the independent-test section: every test recording through the frozen model.

    python3 report/compute_test.py      -> report/test_results.json

Per recording: share of output labels (live classifier, frozen model, same gate and
smoothing as the live view), median measured range, median SNR, and for empty rooms the
largest 4-frame window SNR (what the gate sees). For scenes with two moving targets also
the share of frames in which the strongest moving cell lay at the nearer target.
"""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmwd.classify import DETECT_SNR_DB, LiveClassifier  # noqa: E402
from mmwd.config import parse_cfg  # noqa: E402
from mmwd.freeze import verify  # noqa: E402
from mmwd.session import load_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = verify()          # stops if the frozen files or the model changed
LABELS = ("empty", "unsure", "human", "fan", "drone")
# two-target scenes: range span (m) of the fan or drone, from the single-target checks
# just before (T03 fan at 1.05-1.17 m, T13c drone at 0.82-0.94 m); the person was further
# away (T11, T13) or nearer (T12) and never inside this span
OBJECT_RANGE = {"T11_fan_human_1p5m_0deg": (1.0, 1.25), "T12_fan_human_0p7m_0deg": (1.0, 1.25),
                "T13_drone_human_1p3m_0deg": (0.75, 1.0)}


FROZEN_CFG_PATH = ROOT / "configs" / "md_1tx.cfg"
FROZEN_PARAMS = parse_cfg(FROZEN_CFG_PATH)


def cfg_lines(text):
    """Command lines of a configuration, without comments and blank lines."""
    return [" ".join(l.split()) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("%")]


def run(path, t_max=None):
    s = load_session(path)
    if s.params != FROZEN_PARAMS or cfg_lines(s.meta.get("cfg", "")) != cfg_lines(FROZEN_CFG_PATH.read_text()):
        raise ValueError("chirp configuration differs from the frozen one")
    if len(s.rd) == 0:
        raise ValueError("no range-Doppler frames")
    clf = LiveClassifier(MODEL, s.params)
    lab, feats, ts = [], [], []
    for t, rd in zip(s.t, s.rd):
        if t_max is not None and t >= t_max:
            break
        out, _, f = clf.update(rd)
        feats.append(f)
        ts.append(float(t))
        if out is not None:
            lab.append(str(out))
    c = collections.Counter(lab)
    n = max(len(lab), 1)
    snr = np.array([f["snr_db"] for f in feats])
    rng = np.array([f["_range_m"] for f in feats])
    win = np.convolve(snr, np.ones(4) / 4, "valid") if len(snr) >= 4 else snr
    return s, {
        "frames": len(lab),
        "seconds": round(ts[-1] - ts[0] + 0.25, 1) if ts else 0.0,
        "labels": {k: round(c[k] / n, 3) for k in LABELS},
        "range_median_m": round(float(np.median(rng)), 2),
        "range_p10_p90_m": [round(float(np.percentile(rng, 10)), 2), round(float(np.percentile(rng, 90)), 2)],
        "snr_median_db": round(float(np.median(snr)), 1),
        "window_snr_max_db": round(float(win.max()), 1),
        "note": s.meta.get("note", ""),
    }, rng


out = {"gate_db": DETECT_SNR_DB, "model": MODEL.name, "recordings": {}}
for p in sorted(glob.glob(str(ROOT / "data_test" / "*.npz"))) + \
         sorted(glob.glob(str(ROOT / "data_test_excluded" / "*.npz"))):
    name = Path(p).stem
    try:
        s, r, rng = run(p)
    except ValueError as e:  # T06x: the connection was lost before the first frame (or a wrong configuration)
        out["recordings"][name] = {"error": str(e)}
        continue
    if name in OBJECT_RANGE:
        lo, hi = OBJECT_RANGE[name]
        r["object_share"] = round(float(((rng >= lo) & (rng <= hi)).mean()), 2)
    out["recordings"][name] = r

# T08: the drone fell off the pole near the end; the moment is taken from the data
# (the measured range leaves the bin it held since the start), not from the labels.
s, full, rng = run(ROOT / "data_test" / "T08_drone_0p5m_0deg.npz")
mode = np.bincount(np.round(rng / s.params.range_bin_m).astype(int)).argmax() * s.params.range_bin_m
left = np.abs(rng - mode) > 0.06
if not left.any():
    raise SystemExit("T08: the range never leaves its bin, so there is no fall to cut at")
t_fall = float(s.t[np.argmax(left)])   # first frame outside the bin (a proxy for the fall)
_, pre, _ = run(ROOT / "data_test" / "T08_drone_0p5m_0deg.npz", t_max=t_fall)
out["T08_fall_s"] = round(t_fall, 1)
out["T08_before_fall"] = pre

# drone on a table (the propeller recordings, never used for training)
out["table_mount"] = {}
for p in sorted(glob.glob(str(ROOT / "data_prop" / "*prop_ok*.npz"))):
    out["table_mount"][Path(p).stem] = run(p)[1]

Path(ROOT / "report" / "test_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k, v in out["recordings"].items():
    if "labels" in v:
        print(f"{k:40s} {v['frames']:4d}  " + " ".join(f"{100*v['labels'][l]:4.0f}" for l in LABELS),
              f" R {v['range_median_m']:.2f} ({v['range_p10_p90_m'][0]:.2f}-{v['range_p10_p90_m'][1]:.2f})",
              f"snr {v['snr_median_db']:5.1f} win {v['window_snr_max_db']:5.1f}", v.get("object_share", ""))
    else:
        print(k, v)
print("T08 fall at", out["T08_fall_s"], "s; before:", out["T08_before_fall"]["labels"], out["T08_before_fall"]["frames"])
for k, v in out["table_mount"].items():
    print(k, v["labels"], v["range_median_m"])
