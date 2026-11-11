import sys, glob, json, io, contextlib, collections
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import tempfile, os; TMP_MODEL = os.path.join(tempfile.mkdtemp(prefix='mmwd_'), 'cv_model.joblib')  # one folder per run
import numpy as np, joblib
from sklearn.model_selection import GroupKFold
from mmwd.session import load_session
from mmwd.features import frame_features
from mmwd.classify import build_dataset, make_model, LiveClassifier, DETECT_SNR_DB
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from _ids import rid
CLS=('empty','drone','human','fan')
paths=sorted(p for p in glob.glob('data/*.npz') if load_session(p).label in CLS)
with contextlib.redirect_stdout(io.StringIO()):
    X,y,g,names=build_dataset(paths)
pred=np.empty_like(y); fold_acc=[]
for tr,te in GroupKFold(n_splits=5).split(X,y,g):
    pred[te]=make_model().fit(X[tr],y[tr]).predict(X[te]); fold_acc.append(float((pred[te]==y[te]).mean()))
print('fold acc',np.round(fold_acc,3),'mean %.3f std %.3f'%(np.mean(fold_acc),np.std(fold_acc)))
# cluster bootstrap over recordings
rng=np.random.default_rng(0); groups=np.unique(g); accs=[]; drec=[]
for _ in range(5000):
    gs=rng.choice(groups,len(groups),replace=True)
    idx=np.concatenate([np.where(g==k)[0] for k in gs]); accs.append((pred[idx]==y[idx]).mean())
    d=idx[y[idx]=='drone']; drec.append((pred[d]=='drone').mean() if len(d) else np.nan)
print('bootstrap acc 95%% CI %.3f-%.3f'%tuple(np.percentile(accs,[2.5,97.5])), ' drone recall CI %.3f-%.3f'%tuple(np.nanpercentile(drec,[2.5,97.5])))
print('drone recall %.3f'%((pred[y=='drone']=='drone').mean()), 'precision drone %.3f'%((y[pred=='drone']=='drone').mean()))
# end-to-end windows in drone recordings
tot=0; hit=0; per={}
dp=[p for p in paths if load_session(p).label=='drone']
pos=0
for gi,nm in zip(sorted(set(g)),names): pass
for p in dp:
    s=load_session(p); f=[frame_features(rd,s.params) for rd in s.rd]
    w=np.array([np.mean([x['snr_db'] for x in f[i:i+4]]) for i in range(0,len(f)-3,2)])
    nm=p.split('/')[-1]
    k=[i for i,n in enumerate(names) if n==nm]
    c=int((pred[(g==sorted(set(g))[k[0]])]=='drone').sum()) if k else 0
    per[rid(nm)]=(len(w),int((w>DETECT_SNR_DB).sum()),c); tot+=len(w); hit+=c
print('end-to-end drone windows', hit, tot, '%.3f'%(hit/tot)); print(per)
hang=['R18','R19','R14','R12']
a=sum(per[h][2] for h in hang); b=sum(per[h][0] for h in hang); print('hanging <=1.2m e2e %d/%d %.3f'%(a,b,a/b))
# latency: frames from start until first 'drone' (final) for tethered drone recordings, LOO
lat=[]
for p in dp:
    nm=rid(p)
    if nm not in hang: continue
    with contextlib.redirect_stdout(io.StringIO()):
        Xo,yo,go,_=build_dataset([q for q in paths if q!=p])
    m=make_model().fit(Xo,yo); joblib.dump({'model':m,'win':4},TMP_MODEL)
    s=load_session(p)
    for mode,kw in {'raw':dict(smooth=1,min_conf=0,drone_conf=0),'final':dict(smooth=8,min_conf=0.6,drone_conf=0.7)}.items():
        c=LiveClassifier(TMP_MODEL,s.params,**kw); labs=[c.update(rd)[0] for rd in s.rd]
        first=next((i for i,l in enumerate(labs) if l=='drone'),None); lat.append((nm,mode,first))
print('first drone frame index (4 fps):',lat)
