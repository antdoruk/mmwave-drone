import sys,glob,numpy as np,json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from mmwd.session import load_session
from mmwd.features import frame_features, target_bin
K=2.0
L=lambda n: load_session(glob.glob('data/*'+n+'*')[0])
empty=['R01','R02','R10','R27','R28']
drones={'0.6 m':'R18','1.2 m':'R14','1.9 m':'R13'}
def win_stats(s):
    snr=np.array([frame_features(rd,s.params)['snr_db'] for rd in s.rd])
    return np.array([snr[i:i+4].mean() for i in range(0,len(snr)-3,2)])
E=np.concatenate([win_stats(L(n)) for n in empty]); D={k:win_stats(L(v)) for k,v in drones.items()}
th=np.linspace(0,20,201)
res={'th':th.tolist(),'pfa':[(E>t).mean() for t in th],'pd':{k:[(v>t).mean() for t in th] for k,v in D.items()}}
nfr=sum(len(L(n).rd) for n in empty)
print('empty windows',len(E),'frames',nfr,'max',E.max().round(2),'99pct',np.percentile(E,99).round(2),' rule-of-three bound (non-overlapping windows %d): %.3f'%(nfr//4,3/(nfr//4)))
for k,v in D.items(): print(k,'window mean SNR median %.1f, Pd at 8: %.2f'%(np.median(v),(v>8).mean()))
# frame averaging: power-average N consecutive maps, statistic = peak moving cell at target range minus median (true dB)
def avg_stat(s,N):
    rd=np.asarray(s.rd); p=s.params; nd=p.doppler_bins; mv=np.abs(np.arange(nd)-nd//2)>1
    out=[]
    for i in range(0,len(rd)-N+1,N):
        m=10*np.log10(np.mean(10**(K*rd[i:i+N]/10),axis=0))   # true dB, power average
        rb=target_bin(m/K,p); out.append((m[rb,mv]-np.median(m)).max())
    return np.array(out)
avg={}
for N in [1,2,4,8,16]:
    e=np.concatenate([avg_stat(L(n),N) for n in empty]); d=avg_stat(L(drones['1.9 m']),N); d2=avg_stat(L(drones['1.2 m']),N)
    thr=e.max()
    avg[N]=dict(empty_max=float(thr),empty_med=float(np.median(e)),drone_med=float(np.median(d)),pd=float((d>thr).mean()),n_drone=len(d),n_empty=len(e),d12_med=float(np.median(d2)))
    print(f'N={N:2d} ({N/4:.2f} s): empty median {np.median(e):5.1f} max {thr:5.1f} | drone 1.9 m median {np.median(d):5.1f}  Pd above empty max {avg[N]["pd"]:.2f} ({len(d)} looks) | drone 1.2 m median {np.median(d2):.1f}')
res['avg']=avg
json.dump(res,open('report/det_results.json','w'))
