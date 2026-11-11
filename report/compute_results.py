"""Recompute every number used in the report from the recordings, with the current code."""
import sys, glob, json, io, contextlib, collections
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import tempfile, os; TMP_MODEL = os.path.join(tempfile.mkdtemp(prefix='mmwd_'), 'cv_model.joblib')  # one folder per run
import numpy as np, joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from mmwd.session import load_session
from mmwd.features import frame_features, window_feature_names
from mmwd.classify import build_dataset, make_model, LiveClassifier, DETECT_SNR_DB, MOVING, detection_check
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _ids import rid

CLS = ('empty', 'drone', 'human', 'fan')
paths = sorted(p for p in glob.glob('data/*.npz') if load_session(p).label in CLS)
out = {}

# dataset table
rows = []
for p in paths:
    s = load_session(p)
    feats = [frame_features(rd, s.params) for rd in s.rd]
    a = lambda k: np.array([f[k] for f in feats])
    snr = a('snr_db'); w = np.array([snr[i:i+4].mean() for i in range(0, len(snr)-3, 2)])
    rows.append(dict(file=rid(p), label=s.label, dist=s.meta.get('distance_m'), note=s.meta.get('note', ''),
                     frames=int(len(s.rd)), seconds=float(len(s.rd)/s.params.fps),
                     det=float((w > DETECT_SNR_DB).mean()), rng=float(np.median(a('_range_m'))),
                     snr=float(np.median(snr)), skirt=float(np.median(a('skirt_db'))), occ=float(np.median(a('occupancy'))),
                     hv=float(np.median(a('high_v_frac'))), win_snr_max=float(w.max())))
out['recordings'] = rows

# stage 2 grouped CV, same as train.py
with contextlib.redirect_stdout(io.StringIO()):
    X, y, g, names = build_dataset(paths)
classes = [c for c in MOVING if c in set(y)]
pred = np.empty_like(y)
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
cm = confusion_matrix(y, pred, labels=classes)
out['cv5'] = dict(acc=float((pred == y).mean()), classes=classes, cm=cm.tolist(), n_windows=int(len(y)), n_recordings=len(names),
                  per_class={c: int((y == c).sum()) for c in classes})
perrec = {}
for gi, nm in zip(sorted(set(g)), names):
    sel = g == gi
    perrec[rid(nm)] = dict(label=str(y[sel][0]), n=int(sel.sum()), acc=float((pred[sel] == y[sel]).mean()))
out['cv5_per_recording'] = perrec

# leave-one-recording-out replay through the live classifier, raw and smoothed
tot = collections.defaultdict(collections.Counter); live_rows = {}
for p in paths:
    with contextlib.redirect_stdout(io.StringIO()):
        Xo, yo, go, _ = build_dataset([q for q in paths if q != p])
    m = make_model().fit(Xo, yo)
    joblib.dump({'model': m, 'win': 4}, TMP_MODEL)
    s = load_session(p); res = {}
    for mode, kw in {'raw': dict(smooth=1, min_conf=0.0, drone_conf=0.0), 'final': dict(smooth=8, min_conf=0.6, drone_conf=0.7)}.items():
        c = LiveClassifier(TMP_MODEL, s.params, **kw)
        lab = [c.update(rd)[0] for rd in s.rd]; lab = [l for l in lab if l is not None]
        cnt = collections.Counter(lab); tot[(s.label, mode)] += cnt
        res[mode] = {k: v/len(lab) for k, v in cnt.items()}
    live_rows[rid(p)] = dict(label=s.label, **res)
out['replay_per_recording'] = live_rows
out['replay_totals'] = {f'{k[0]}|{k[1]}': {kk: vv/sum(v.values()) for kk, vv in v.items()} for k, v in tot.items()}

# final model and feature importance
final = make_model().fit(X, y)
imp = sorted(zip(window_feature_names(), final.feature_importances_), key=lambda t: -t[1])
out['importance'] = [(n, float(v)) for n, v in imp]
json.dump(out, open('report/results.json', 'w'), indent=1)
print('cv5 acc', out['cv5']['acc'], out['cv5']['cm'])
for k, v in sorted(out['replay_totals'].items()): print(k, {kk: round(vv*100, 1) for kk, vv in sorted(v.items())})
print('top importance', out['importance'][:8])
