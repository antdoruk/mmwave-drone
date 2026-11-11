"""Independent test: replay new recordings through the frozen model, unchanged.

    python3 scripts/eval_test.py data_test/*.npz

First checks that the classification code and the model are byte-identical to the
frozen manifest, then plays every recording frame by frame through LiveClassifier
(same gate, smoothing and thresholds as the live view) and prints the share of
frames per output label. Nothing here is tuned on the test data.

A recording is not scored, and the exit code is 2, if its chirp configuration differs from
the frozen one (configs/md_1tx.cfg: the stored command lines and the derived
parameters must both match) or it has too few range-Doppler frames for one window.
"""
import argparse
import collections
import sys
from pathlib import Path

import _path  # noqa: F401

from mmwd.classify import DETECT_SNR_DB, LiveClassifier
from mmwd.config import parse_cfg
from mmwd.freeze import verify
from mmwd.session import load_session

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("files", nargs="+")
ap.add_argument("--frozen", default=None, help="frozen folder (default: models/frozen)")
args = ap.parse_args()

model = verify(args.frozen)
FROZEN_CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "md_1tx.cfg"
FROZEN_PARAMS = parse_cfg(FROZEN_CFG_PATH)


def cfg_lines(text):
    """Command lines of a configuration, without comments and blank lines."""
    return [" ".join(l.split()) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("%")]


FROZEN_CFG = cfg_lines(FROZEN_CFG_PATH.read_text())
print(f"\ngate {DETECT_SNR_DB} dB, model {model.name}\n")
LABELS = ("empty", "unsure", "human", "fan", "drone")
print(f"{'recording':34s} {'frames':>6s} " + " ".join(f"{k:>7s}" for k in LABELS) + "  note")
unscored = 0
for f in args.files:
    s = load_session(f)
    why = ("chirp configuration differs from the frozen one"
           if s.params != FROZEN_PARAMS or cfg_lines(s.meta.get("cfg", "")) != FROZEN_CFG
           else "no range-Doppler frames" if len(s.rd) == 0 else None)
    if why:
        print(f"{Path(f).stem:34s} not scored: {why}")
        unscored += 1
        continue
    clf = LiveClassifier(model, s.params)
    out = [clf.update(rd)[0] for rd in s.rd]
    out = [o for o in out if o is not None]          # first frames fill the window
    if not out:
        print(f"{Path(f).stem:34s} not scored: too few frames for one window")
        unscored += 1
        continue
    c = collections.Counter(out)
    n = len(out)
    print(f"{Path(f).stem:34s} {len(out):6d} " + " ".join(f"{100 * c[k] / n:6.0f}%" for k in LABELS)
          + f"  {s.meta.get('note', '')}")
if unscored:
    print(f"\n{unscored} recording(s) not scored")
    sys.exit(2)
