import sys,glob,numpy as np,json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from mmwd.session import load_session
from mmwd.features import target_bin
K=2.0
L=lambda n: load_session(glob.glob('data/*'+n+'*')[0])
empty=['R01','R02','R10','R27','R28']
d19=L('R13'); p=d19.params; nd=p.doppler_bins; mv=np.abs(np.arange(nd)-nd//2)>1
rbs=[target_bin(rd,p) for rd in d19.rd]; vals,c=np.unique(rbs,return_counts=True); RB=int(vals[np.argmax(c)])
print('drone 1.9 m modal bin',RB,RB*p.range_bin_m)
def stat(s,N,rb):
    rd=np.asarray(s.rd); out=[]
    for i in range(0,len(rd)-N+1,N):
        m=10*np.log10(np.mean(10**(K*rd[i:i+N]/10),axis=0))
        out.append((m[rb-1:rb+2][:,mv]-np.median(m)).max())   # peak in the 3 bins around the drone range
    return np.array(out)
res={}
for N in [1,2,4,8,16,32]:
    e=np.concatenate([stat(L(n),N,RB) for n in empty]); d=stat(d19,N,RB)
    thr=e.max(); pd=(d>thr).mean()
    res[N]=dict(e_med=float(np.median(e)),e_max=float(thr),d_med=float(np.median(d)),pd=float(pd),nd=len(d),ne=len(e),margin=float(np.median(d)-thr))
    print(f'N={N:2d} ({N/4:5.2f} s) empty@1.9m median {np.median(e):5.2f} max {thr:5.2f} | drone median {np.median(d):5.2f} | median-margin {np.median(d)-thr:+5.2f} dB | Pd {pd:.2f} ({len(d)} looks, {len(e)} empty looks)')
json.dump({'rb':RB,'res':res},open('report/avg_results.json','w'))
