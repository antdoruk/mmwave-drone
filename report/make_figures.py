"""All report figures from the recordings. Run from the repo root: python3 report/make_figures.py"""
import sys, glob, json, io, contextlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mmwd.session import load_session
from mmwd.features import frame_features, target_bin, window_features
from mmwd.sim import Simulator, Drone, Fan, Static

OUT = 'report/figs/'
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 9.5, 'axes.titleweight': 'normal',
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 220})
COL = {'empty': '#8c8c8c', 'drone': '#c0392b', 'human': '#2b6cb0', 'fan': '#6b8e23', 'unsure': '#e0b84c'}
D = 'data/'
R = json.load(open('report/results.json'))


def S(name):
    return load_session(glob.glob(D + name + '*.npz')[0])


def avg_db(rd):
    return 10 * np.log10(np.mean(10 ** (np.asarray(rd) / 10), axis=0))


def spec_at_target(s):
    out = []
    for rd in s.rd:
        rb = target_bin(rd, s.params)
        out.append(rd[rb] - np.median(rd))
    return 10 * np.log10(np.mean(10 ** (np.array(out) / 10), axis=0))


# 1. averaged range-Doppler maps, one per class -------------------------------------------
cases = [('Empty room', 'R02'), ('Person waving arms, 0.7 m', 'R21'),
         ('Fan, setting 3, 1 m', 'R16'), ('Drone tethered, 0.5 m', 'R18')]
fig, axs = plt.subplots(2, 2, figsize=(6.2, 5.0), sharey=True, sharex=True); axs = axs.ravel()
for ax, (t, n) in zip(axs, cases):
    s = S(n); p = s.params; v = p.velocity_axis(); r = p.range_axis()
    dv, dr = v[1] - v[0], r[1] - r[0]
    m = avg_db(s.rd); m = m - np.median(m)
    im = ax.imshow(m, aspect='auto', origin='lower', extent=[v[0] - dv / 2, v[-1] + dv / 2, r[0] - dr / 2, r[-1] + dr / 2],   # outer edges of the cells
                   vmin=0, vmax=28, cmap='viridis')
    ax.set_ylim(0, 3.2); ax.set_title(t); ax.set_xlabel('radial velocity (m/s)')
    ax.set_xticks([-30, -15, 0, 15, 30])
for a in axs[:2]: a.set_xlabel('')
axs[0].set_ylabel('range (m)'); axs[2].set_ylabel('range (m)')
cb = fig.colorbar(im, ax=axs, shrink=0.95, pad=0.015); cb.set_label('dB above map median')
fig.savefig(OUT + 'rd_maps.pdf')
plt.close(fig)

# 2. Doppler spectra at the target range -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 2.7))
for n, lab, c in [('R18', 'drone, 0.5 m', COL['drone']), ('R16', 'fan, setting 3, 1 m', COL['fan']),
                  ('R21', 'person waving, 0.7 m', COL['human']), ('R27', 'empty room', COL['empty'])]:
    s = S(n); v = s.params.velocity_axis()
    ax.plot(v, spec_at_target(s), color=c, lw=1.2, label=lab)
ax.axvspan(10, 25, color='0.85', zorder=0); ax.axvspan(-25, -10, color='0.85', zorder=0)
ax.text(17.5, 21.5, 'skirt band', ha='center', fontsize=8, color='0.35')
ax.set_xlabel('radial velocity (m/s)'); ax.set_ylabel('dB above map median')
ax.set_xlim(-36, 36); ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.2)
fig.savefig(OUT + 'spectra.pdf'); plt.close(fig)

# 3. drone signature against distance ------------------------------------------------------------
rows = {r['file']: r for r in R['recordings']}
drone = [('R18', 'hanging'), ('R12', 'hanging, swinging'), ('R19', 'hanging, 46 deg off axis'),
         ('R14', 'hanging'), ('R13', 'hanging')]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
for n, note in drone:
    r = rows[n]; mk = 'o' if 'off axis' not in note else 's'
    axs[0].plot(r['rng'], r['snr'], mk, color=COL['drone'], ms=5)
    axs[1].plot(r['rng'], r['skirt'], mk, color=COL['drone'], ms=5)
fan_sk = [rows[k]['skirt'] for k in rows if rows[k]['label'] == 'fan']
hum_sk = [rows[k]['skirt'] for k in rows if rows[k]['label'] == 'human' and rows[k]['det'] > 0.9]
axs[1].axhspan(min(fan_sk + hum_sk), max(fan_sk + hum_sk), color='0.88', zorder=0)
axs[1].text(1.9, max(fan_sk + hum_sk) + 0.15, 'range of fan and person\nrecordings', ha='right', va='bottom', fontsize=7.5, color='0.35')
axs[0].axhline(8, color='k', ls='--', lw=0.8); axs[0].text(1.95, 8.3, 'detection gate', ha='right', fontsize=7.5)
axs[0].set_ylabel('median peak SNR (dB)'); axs[1].set_ylabel('median skirt level (dB)')
for a in axs: a.set_xlabel('measured range (m)'); a.set_xlim(0.4, 2.0); a.grid(alpha=0.2)
axs[1].set_ylim(0, 8.5)
fig.tight_layout(); fig.savefig(OUT + 'distance.pdf'); plt.close(fig)

# 4. feature space, one dot per 1 s window -------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.3))
for p in sorted(glob.glob(D + '*.npz')):
    s = load_session(p)
    if s.label not in COL or s.label == 'empty' and False:
        continue
    if s.label not in ('drone', 'human', 'fan', 'empty'):
        continue
    f = [frame_features(rd, s.params) for rd in s.rd]
    for i in range(0, len(f) - 3, 2):
        w = f[i:i + 4]
        if s.label != 'empty' and np.mean([x['snr_db'] for x in w]) <= 8:
            continue
        ax.plot(np.mean([x['occupancy'] for x in w]), np.mean([x['skirt_db'] for x in w]), '.',
                color=COL[s.label], ms=2.5, alpha=0.5)
for k in ('drone', 'human', 'fan', 'empty'):
    ax.plot([], [], 'o', color=COL[k], label=k, ms=4)
ax.set_xlabel('occupancy (share of Doppler bins > floor + 6 dB)'); ax.set_ylabel('skirt level (dB)')
ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.2)
fig.savefig(OUT + 'feature_space.pdf'); plt.close(fig)

# 5. confusion matrix ---------------------------------------------------------------------------
cm = np.array(R['cv5']['cm']); cls = R['cv5']['classes']
fig, ax = plt.subplots(figsize=(3.0, 2.6))
cmn = cm / cm.sum(1, keepdims=True)
ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(len(cls)):
    for j in range(len(cls)):
        ax.text(j, i, f'{cm[i, j]}\n{100 * cmn[i, j]:.0f}%', ha='center', va='center', fontsize=8,
                color='w' if cmn[i, j] > 0.5 else 'k')
ax.set_xticks(range(len(cls)), cls); ax.set_yticks(range(len(cls)), cls)
ax.set_xlabel('predicted'); ax.set_ylabel('true')
for sp in ax.spines.values(): sp.set_visible(False)
fig.savefig(OUT + 'confusion.pdf'); plt.close(fig)

# 6. replay through the live classifier, raw vs final ------------------------------------------------
tot = R['replay_totals']
order = ['empty', 'drone', 'human', 'fan']
outs = ['empty', 'drone', 'human', 'fan', 'unsure']
fig, ax = plt.subplots(figsize=(6.4, 2.4))
y = 0; yt = []; yl = []
for c in order:
    for mode, lab in (('raw', 'per window'), ('final', 'smoothed')):
        d = tot[f'{c}|{mode}']; left = 0
        for o in outs:
            w = 100 * d.get(o, 0)
            if w > 0:
                ax.barh(y, w, left=left, color=COL[o], edgecolor='white', lw=0.5)
                if w >= 7: ax.text(left + w / 2, y, f'{w:.0f}', ha='center', va='center', fontsize=7, color='white' if o != 'unsure' else 'k')
                left += w
        yt.append(y); yl.append(f'{c}: {lab}'); y += 1
    y += 0.5
ax.set_yticks(yt, yl); ax.invert_yaxis(); ax.set_xlabel('share of frames (%)'); ax.set_xlim(0, 100)
ax.legend(handles=[Patch(color=COL[o], label=o) for o in outs], ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
fig.savefig(OUT + 'replay.pdf'); plt.close(fig)

# 7. live session timeline (screen recording, earlier model version, no smoothing) ---------
seq = {}
def put(a, b, lab):
    for t in range(a, b + 1): seq[t] = lab
put(43, 60, 'empty'); put(61, 75, 'human'); seq[64] = 'empty'; seq[76] = 'fan'; put(77, 78, 'drone'); put(79, 80, 'human')
put(81, 83, 'drone'); put(84, 106, 'fan'); seq[107] = 'human'; put(108, 110, 'drone'); put(111, 113, 'empty'); put(114, 115, 'human')
seq[116] = 'drone'; put(117, 120, 'empty'); seq[121] = 'human'; put(122, 125, 'empty'); seq[126] = 'drone'; put(127, 129, 'empty')
seq[130] = 'human'; put(131, 138, 'drone'); put(139, 141, 'empty'); seq[142] = 'fan'; put(143, 144, 'drone'); seq[145] = 'human'
put(146, 147, 'empty'); put(148, 152, 'drone'); put(153, 156, 'empty')
truth = [(43, 60, 'empty room'), (61, 75, 'walking'), (76, 83, 'A'), (84, 106, 'fan on'),
         (107, 130, 'setting up drone (B)'), (131, 152, 'drone on (C)'), (153, 156, '')]
fig, ax = plt.subplots(figsize=(7.0, 1.55))
for t, lab in seq.items():
    ax.barh(0, 1, left=t - 43, color=COL[lab], height=0.8)
for a, b, txt in truth:
    ax.barh(1, b - a + 1, left=a - 43, color='white', edgecolor='0.4', height=0.8, lw=0.6)
    ax.text((a + b + 1) / 2 - 43, 1, txt, ha='center', va='center', fontsize=6.5)
ax.set_yticks([0, 1], ['label shown', 'what happened']); ax.set_xlabel('time (s)'); ax.set_xlim(0, 114)
ax.legend(handles=[Patch(color=COL[k], label=k) for k in ('empty', 'drone', 'human', 'fan')], ncol=4, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=7)
for sp in ('left',): ax.spines[sp].set_visible(False)
fig.savefig(OUT + 'live_timeline.pdf'); plt.close(fig)

# 8. simulator against measurement, drone ----------------------------------------------------------
s = S('R18'); p = s.params; v = p.velocity_axis()
meas = spec_at_target(s)
sim = Simulator(p, seed=3)
frames = [np.fft.fftshift(rd, axes=1) for _, rd, _ in sim.frames([Drone(hub=(0, 0.59, 0), seed=3)], 24)]
sim_spec = []
for rd in frames:
    rb = target_bin(rd, p); sim_spec.append(rd[rb] - np.median(rd))
sim_spec = 10 * np.log10(np.mean(10 ** (np.array(sim_spec) / 10), axis=0))
fig, ax = plt.subplots(figsize=(6.4, 2.4))
ax.plot(v, meas, color=COL['drone'], lw=1.2, label='measured, 0.5 m')
ax.plot(v, sim_spec - (np.median(sim_spec) - np.median(meas)), color='k', lw=1.0, ls='--', label='simulated, same chirp, 500 rev/s')
ax.set_xlabel('radial velocity (m/s)'); ax.set_ylabel('dB above map median'); ax.set_xlim(-36, 36)
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.2)
fig.savefig(OUT + 'sim_vs_meas.pdf'); plt.close(fig)

# 9. propeller damage ------------------------------------------------------------------------------
P = 'data_prop/'
def hv(n):
    s = load_session(glob.glob(P + n + '*.npz')[0]); v = s.params.velocity_axis(); t = np.asarray(s.t) - s.t[0]
    Sp = np.array([rd[target_bin(rd, s.params)] - np.median(rd) for rd in s.rd])
    return s, v, t, Sp
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={'width_ratios': [1.35, 1]})
s1, v, t1, A = hv('P06'); s2, _, t2, B = hv('P07')
m1 = t1 >= t1[-1] - 6.5; m2 = t2 >= t2[-1] - 6.5
axs[0].plot(v, 10 * np.log10(np.mean(10 ** (A[m1] / 10), 0)), color='k', lw=1.1, label='new propeller')
axs[0].plot(v, 10 * np.log10(np.mean(10 ** (B[m2] / 10), 0)), color=COL['drone'], lw=1.1, label='same propeller, half of one blade cut')
axs[0].axvspan(10, 36, color='0.9', zorder=0); axs[0].axvspan(-36, -10, color='0.9', zorder=0)
axs[0].set_xlim(-36, 36); axs[0].set_xlabel('radial velocity (m/s)'); axs[0].set_ylabel('dB above map median')
axs[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1); axs[0].set_title('0.6 m, last 6.5 s of each recording')
names = [('P01', 'intact'), ('P02', 'intact'), ('P03', '2 mm'), ('P04', '7 mm'),
         ('P05', 'half*')]
vals = []
for n, lab in names:
    s, v, t, Sp = hv(n); vals.append(Sp[:, np.abs(v) > 10].mean())
cols = ['0.3', '0.3', COL['drone'], COL['drone'], '#e8a49b']
axs[1].bar(range(len(vals)), vals, color=cols)
axs[1].set_xticks(range(len(vals)), [l for _, l in names]); axs[1].set_ylabel('mean level, |v| > 10 m/s (dB)')
axs[1].set_title('0.5 m, cut length of one blade')
fig.tight_layout(); fig.savefig(OUT + 'propeller.pdf'); plt.close(fig)
# numbers of the propeller table: high band (|v| > 10 m/s), median skirt, low band (moving, |v| < 10 m/s)
for n in [x for x, _ in names] + ['P06', 'P07']:
    s, v, t, Sp = hv(n)
    m = t >= t[-1] - 6.5 if n in ('P06', 'P07') else np.ones(len(t), bool)
    k = np.abs(np.arange(len(v)) - len(v) // 2)
    sk = np.median([frame_features(rd, s.params)['skirt_db'] for rd, keep in zip(s.rd, m) if keep])
    print(f'propeller {n}: high band {Sp[m][:, np.abs(v) > 10].mean():.2f} dB, skirt {sk:.2f} dB, '
          f'low band {Sp[m][:, (k > 1) & (np.abs(v) < 10)].mean():.1f} dB')
print('figures done')
