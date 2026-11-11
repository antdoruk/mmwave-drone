"""Windows of frames -> feature vectors -> a small, explainable classifier.

Deliberately simple (standardised features + random forest / logistic
regression). With a few hundred windows, a CNN would mostly memorise the
room; a handful of physical features generalise better and each of them has
a physical meaning.

Evaluation is grouped by recording: windows from the same recording never
appear in both training and test folds, otherwise accuracy is inflated by
near-duplicate frames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import frame_features, window_feature_names, window_features
from .session import Session, load_session

CLASSES = ("empty", "drone", "human", "fan")
MOVING = ("drone", "human", "fan")

# Stage 1, detection: a window is "empty" when the mean SNR of its strongest moving
# cell is at or below this. 5 empty-room recordings, with the 0.45 m
# range cut and 1 guard bin: window means at most 7.2 dB. 8.0 dB leaves 0.8 dB margin;
# was 9.0 before the range cut. Re-check if the radar moves to another room.
DETECT_SNR_DB = 8.0


def session_windows(s: Session, win: int = 4, hop: int = 2, min_snr_moving: float | None = None):
    """Windows of `win` frames every `hop` frames. For moving classes, windows whose mean
    SNR is below `min_snr_moving` are dropped: a person who pauses (e.g. turning round) is
    removed by static clutter removal and those frames are physically an empty room."""
    feats = [frame_features(rd, s.params) for rd in s.rd]
    X = []
    for i in range(0, len(feats) - win + 1, hop):
        w = feats[i:i + win]
        if (min_snr_moving is not None and s.label != "empty"
                and np.mean([f["snr_db"] for f in w]) < min_snr_moving):
            continue
        X.append(window_features(w))
    return np.array(X), feats


def build_dataset(paths, win: int = 4, hop: int = 2, labels=MOVING, min_snr_moving: float | None = DETECT_SNR_DB):
    """Stage 2 training set: moving classes only, windows that pass the detection gate."""
    X, y, g, names = [], [], [], []
    for gi, p in enumerate(sorted(paths)):
        s = load_session(p)
        if s.label not in labels:
            print(f"skip {Path(p).name}: label '{s.label}' not in {labels}")
            continue
        Xs, _ = session_windows(s, win, hop, min_snr_moving)
        if len(Xs) == 0:
            continue
        X.append(Xs)
        y += [s.label] * len(Xs)
        g += [gi] * len(Xs)
        names.append(Path(p).name)
    return np.vstack(X), np.array(y), np.array(g), names


def make_model(kind: str = "rf"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0,
                                      class_weight="balanced")
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


class LiveClassifier:
    """Keeps the last `win` frames' features and predicts on every new frame.

    Smoothing: the class probabilities of the last `smooth` predictions are averaged
    (8 frames = 2 s at 4 fps), so one odd window cannot flip the label. If the
    averaged winner is below `min_conf`, the label is "unsure" instead of a guess.
    Brief arm or hand movements right in front of the radar were the main source
    of short false "drone" labels before this (live test).
    "drone" needs more confidence (`drone_conf`) than the other classes: a false drone
    alarm costs more than a late one. 0.7 removed the remaining false drones in
    leave-one-recording-out replays while keeping tethered-drone recordings at 75-98%."""

    def __init__(self, model_path: str | Path, params, win: int = 4, smooth: int = 8,
                 min_conf: float = 0.6, drone_conf: float = 0.7):
        import joblib
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.win = bundle.get("win", win)
        self.params = params
        self.smooth = smooth
        self.min_conf = min_conf
        self.drone_conf = drone_conf
        self.buf = []
        self.probs = []

    def update(self, rd_db: np.ndarray):
        f = frame_features(rd_db, self.params)
        self.buf = (self.buf + [f])[-self.win:]
        if len(self.buf) < self.win:
            return None, None, f
        if np.mean([b["snr_db"] for b in self.buf]) <= DETECT_SNR_DB:
            self.probs = []                     # target gone: start fresh next time
            return "empty", None, f
        x = window_features(self.buf)[None, :]
        self.probs = (self.probs + [self.model.predict_proba(x)[0]])[-self.smooth:]
        p = np.mean(self.probs, axis=0)
        k = int(np.argmax(p))
        need = self.drone_conf if self.model.classes_[k] == "drone" else self.min_conf
        if len(self.probs) < min(3, self.smooth) or p[k] < need:
            return "unsure", float(p[k]), f
        return self.model.classes_[k], float(p[k]), f


__all__ = ["build_dataset", "make_model", "LiveClassifier", "window_feature_names", "CLASSES", "MOVING", "DETECT_SNR_DB", "detection_check"]


def detection_check(paths, win: int = 4, hop: int = 2):
    """Stage 1 on every recording: share of windows above the gate, per recording."""
    rows = []
    for p in sorted(paths):
        s = load_session(p)
        feats = [frame_features(rd, s.params) for rd in s.rd]
        m = [np.mean([f["snr_db"] for f in feats[i:i + win]]) for i in range(0, len(feats) - win + 1, hop)]
        rows.append((Path(p).name, s.label, float(np.mean(np.array(m) > DETECT_SNR_DB)), len(m)))
    return rows
