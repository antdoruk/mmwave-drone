import sys,glob,json,numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
from mmwd.session import load_session
from mmwd.features import target_bin
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.bbox':'tight','savefig.dpi':220,'axes.titlesize':9.5})
OUT='report/figs/'; L=lambda n: load_session(glob.glob('data/*'+n+'*')[0])
RED='#c0392b'
# --- scale: noise statistics + range law
empty=['R01','R02','R10','R27','R28']
x=[]
for n in empty:
    rd=np.asarray(L(n).rd)[:,10:60,:]; nd=rd.shape[2]; m=np.abs(np.arange(nd)-nd//2)>3
    v=rd[:,:,m]; v=v-np.median(v,axis=(1,2),keepdims=True); x.append(v.ravel())
x=np.concatenate(x); sd=x.std()
rng=np.random.default_rng(0); Z=(rng.standard_normal((400000,4))+1j*rng.standard_normal((400000,4)))/np.sqrt(2)
th=np.mean(20*np.log10(np.abs(Z)),axis=1); th=th-np.median(th)
K=th.std()/sd
print('noise std map units %.3f, theory (mean of 4 log-magnitudes) %.3f dB, ratio K=%.2f; skew meas %.2f theory %.2f'%(sd,th.std(),K,((x-x.mean())**3).mean()/sd**3,((th-th.mean())**3).mean()/th.std()**3))
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6))
b=np.linspace(-12,8,81)
ax[0].hist(x,bins=b/2,density=True,histtype='step',color='0.5',lw=1.2,label='map units, as converted')
ax[0].hist(2*x,bins=b,density=True,histtype='step',color='k',lw=1.4,label='map units $\\times$ 2')
ax[0].hist(th,bins=b,density=True,histtype='stepfilled',color=RED,alpha=0.25,label='theory: thermal noise,\nmean of 4 antenna log-magnitudes')
ax[0].set_xlabel('noise cell level minus median (dB)'); ax[0].set_ylabel('density'); ax[0].legend(fontsize=6.8,loc='upper left'); ax[0].set_title('Empty room: noise statistics')
pts=[('R18',0.59),('R14',1.17),('R13',1.87)]
def integ(s,k):
    # noise-subtracted echo of the moving bins at the target row (same estimator as analysis_rcs.py)
    p=s.params; nd=p.doppler_bins; mv=np.abs(np.arange(nd)-nd//2)>1; T=[]; N=[]
    rb=int(np.bincount([target_bin(rd,p) for rd in s.rd]).argmax()); ref=[r for r in range(8,60) if abs(r-rb)>3]
    for rd in s.rd:
        L=10**(k*(rd-np.median(rd))/10); T.append(L[rb,mv].sum()); N.append(L[ref][:,mv].sum(1).mean())
    return 10*np.log10(np.mean(T)-np.mean(N))
R=np.array([r for _,r in pts]); I1=np.array([integ(L(n),1.0) for n,_ in pts]); I2=np.array([integ(L(n),2.0) for n,_ in pts])
rr=np.linspace(0.5,2.1,50)
ax[1].semilogx(R,I2,'o',color='k',label='map units $\\times$ 2'); ax[1].semilogx(R,I1,'s',color='0.6',ms=4,label='map units, as converted')
ax[1].semilogx(rr,I2[0]-40*np.log10(rr/R[0]),'-',color=RED,lw=1,label='$R^{-4}$ (radar equation)')
ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].set_xticks([0.5,0.75,1,1.5,2]); ax[1].set_xticklabels(['0.5','0.75','1','1.5','2'])
ax[1].set_xlabel('range (m)'); ax[1].set_ylabel('integrated drone echo / noise (dB)'); ax[1].legend(fontsize=6.8); ax[1].set_title('Tethered drone: echo against range')
fig.tight_layout(); fig.savefig(OUT+'scale.pdf'); plt.close(fig)
print('integrated x2',I2.round(1),' x1',I1.round(1))
# --- ROC / gate curves and per-cell averaging
D=json.load(open('report/det_results.json')); A=json.load(open('report/avg_results.json'))
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7))
t=np.array(D['th'])
ax[0].plot(2*t,D['pfa'],color='0.4',lw=1.5,label='empty room (false alarm)')
for (k,v),c in zip(D['pd'].items(),['#7f1d1d',RED,'#f0a090']): ax[0].plot(2*t,v,color=c,lw=1.3,label=f'drone, {k}')
ax[0].axvline(16,color='k',ls='--',lw=0.8); ax[0].text(16.5,0.08,'gate',fontsize=7.5)
ax[0].set_xlabel('threshold on 1 s mean of peak SNR (dB)'); ax[0].set_ylabel('share of windows above'); ax[0].legend(fontsize=6.8,loc='upper right',bbox_to_anchor=(1.02,0.95)); ax[0].set_title('Global gate: peak at the row with most moving energy', fontsize=8.5)
N=np.array(sorted(int(k) for k in A['res'])); r=A['res']
ax[1].semilogx(N/4,[r[str(n)]['d_med'] for n in N],'o-',color=RED,label='drone at 1.9 m, median')
ax[1].semilogx(N/4,[r[str(n)]['e_max'] for n in N],'s--',color='0.3',label='empty room, maximum')
ax[1].semilogx(N/4,[r[str(n)]['e_med'] for n in N],'s:',color='0.6',label='empty room, median')
ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].set_xticks(N/4); ax[1].set_xticklabels([f'{n/4:g}' for n in N])
ax[1].set_xlabel('averaging time (s)'); ax[1].set_ylabel('peak level at the drone range (dB)'); ax[1].legend(fontsize=7); ax[1].set_title('Same cells, frames power-averaged')
fig.tight_layout(); fig.savefig(OUT+'detection.pdf'); plt.close(fig)
print('ok')
