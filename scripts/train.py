"""Train the classifier and evaluate it recording by recording.

    python3 scripts/train.py data/*.npz
    python3 scripts/train.py data_sim/*.npz --out models/model_sim.joblib

Evaluation: grouped cross-validation, every recording is held out as a whole.
Writes the confusion matrix and feature importance figures to figures/.
"""
import argparse
from pathlib import Path

import _path  # noqa: F401
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from mmwd.classify import MOVING as CLASSES, build_dataset, make_model  # noqa: E402
from mmwd.features import window_feature_names  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--win", type=int, default=4, help="frames per window (4 = 1 s at 4 fps)")
ap.add_argument("--model", default="rf", choices=["rf", "logreg"])
ap.add_argument("--out", default="models/model.joblib")
ap.add_argument("--tag", default="")
args = ap.parse_args()

from mmwd.classify import DETECT_SNR_DB, detection_check  # noqa: E402

print(f"Stage 1, detection gate: window mean SNR > {DETECT_SNR_DB} dB")
for name, label, frac, n in detection_check(args.files, win=args.win):
    want = "should be 0%" if label == "empty" else "share detected"
    print(f"  {label:>6}  {frac:6.1%} of {n:3d} windows above gate ({want})  {name}")
print()
print("Stage 2, classification of detected windows")
X, y, g, names = build_dataset(args.files, win=args.win)
classes = [c for c in CLASSES if c in set(y)]
print(f"{len(names)} recordings, {len(y)} windows: " +
      ", ".join(f"{c}={int((y == c).sum())}" for c in classes))

n_groups = len(set(g))
k = min(5, n_groups)
pred = np.empty_like(y)
for tr, te in GroupKFold(n_splits=k).split(X, y, g):
    m = make_model(args.model).fit(X[tr], y[tr])
    pred[te] = m.predict(X[te])
acc = (pred == y).mean()
cm = confusion_matrix(y, pred, labels=classes)
print(f"\ngrouped {k}-fold accuracy: {acc:.1%}  (chance ~{1/len(classes):.0%})")
print("rows = truth, columns = prediction\n" + " " * 8 + " ".join(f"{c:>7}" for c in classes))
for c, row in zip(classes, cm):
    print(f"{c:>7} " + " ".join(f"{v:7d}" for v in row))

# per-recording errors, the most useful table when something goes wrong
print("\nrecordings with errors:")
for gi, nm in zip(sorted(set(g)), names):
    sel = g == gi
    err = (pred[sel] != y[sel]).mean()
    if err > 0:
        print(f"  {nm}: {err:.0%} wrong, predicted {dict((str(a), int(b)) for a, b in zip(*np.unique(pred[sel], return_counts=True)))}")

final = make_model(args.model).fit(X, y)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": final, "win": args.win, "features": window_feature_names(), "cv_acc": acc}, args.out)
print(f"\nsaved {args.out}")

Path("figures").mkdir(exist_ok=True)
tag = args.tag or Path(args.out).stem
fig, ax = plt.subplots(figsize=(4.5, 4))
cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="w" if cmn[i, j] > 0.5 else "k")
ax.set_xticks(range(len(classes)), classes)
ax.set_yticks(range(len(classes)), classes)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"held-out recordings, acc {acc:.0%}")
fig.tight_layout()
fig.savefig(f"figures/confusion_{tag}.png", dpi=150)

if args.model == "rf":
    imp = final.feature_importances_
    order = np.argsort(imp)[::-1][:10]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([window_feature_names()[i] for i in order][::-1], imp[order][::-1])
    ax.set_title("what the classifier relies on")
    fig.tight_layout()
    fig.savefig(f"figures/importance_{tag}.png", dpi=150)
print(f"figures/confusion_{tag}.png written")
