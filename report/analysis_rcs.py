"""Radar equation check and radar cross section (RCS) of the moving parts.

    python3 report/analysis_rcs.py      -> report/rcs_results.json

Echo estimator (noise-subtracted): the target row is the range row the target occupies most
often in the recording (fixed for the whole recording, so no per-frame selection of noise
peaks). Each frame is converted to linear units relative to the map median with a scale K
(true dB per map unit), the moving Doppler bins of that row are summed, and the same sum
averaged over reference rows far from the target is subtracted; both are averaged over the
recording before the subtraction. Choosing the row frame by frame instead (strongest row in
each frame) gives the same values within 0.1 dB for the drone recordings. Neither choice is
free of selection bias: in an empty room the modal row is itself a noise peak (or a weak
distant echo), and the same estimator returns an apparent echo of about 10 to 18 dB at K = 2
(empty_control in the output). For the drone the row is set by the target (the same row in
78 to 100 % of frames), so the bias is smaller, but it cannot be excluded for the weakest
point. An earlier version clipped (lin - 1) at zero per bin, which kept positive noise
fluctuations and biased the weakest (most distant) point upwards by 1.6-3.4 dB.

The three training recordings were used to choose K, so they cannot validate it. Recordings
from the independent test (T09, T10 on the same pole; U05, U04 on a sofa arm in another room)
give a separate check.

The absolute RCS depends on the datasheet EIRP and EINF (typical values), on the window
loss, on how the firmware combines the antennas and on K. It is a model-dependent
estimate, not a calibrated measurement.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmwd.session import load_session  # noqa: E402
from mmwd.features import target_bin  # noqa: E402
from _ids import rid  # noqa: E402

LAM, TOBS, KT0 = 4.895e-3, 128 * 12.8e-6, 4.0e-21
EIRP, EINF, WIN_LOSS = 10 ** (16 / 10) * 1e-3, 10 ** (9 / 10), 3.5


def echo_db(name, K, folder='data', rows=(0,)):
    """Noise-subtracted echo of the moving parts at the target row, true dB over the median."""
    s = load_session(glob.glob(folder + '/*' + name + '*')[0] if not name.endswith('.npz') else name)
    p = s.params
    nd = p.doppler_bins
    mv = np.abs(np.arange(nd) - nd // 2) > 1
    rbs = [target_bin(rd, p) for rd in s.rd]
    rb = int(np.bincount(rbs).argmax())                     # fixed row for the recording
    ref = [r for r in range(8, 60) if abs(r - rb) > 3]
    T, N = [], []
    for rd in s.rd:
        L = 10 ** (K * (rd - np.median(rd)) / 10)
        T.append(sum(L[rb + d, mv].sum() for d in rows))
        N.append(len(rows) * L[ref][:, mv].sum(1).mean())
    sig = np.mean(T) - np.mean(N)
    return (float(10 * np.log10(sig)) if sig > 0 else float('nan')), float(rb * p.range_bin_m)


base = 10 * np.log10(EIRP * LAM ** 2 * TOBS / ((4 * np.pi) ** 3 * KT0 * EINF)) - WIN_LOSS
print(f'expected SNR for 1 m^2 at 1 m: {base:.1f} dB (after {WIN_LOSS} dB window loss)')

# the three tethered-drone recordings used for the R^-4 test
DRONE = [('R18', 'drone tethered, nominal 0.5 m'), ('R14', 'drone tethered, nominal 1 m'),
         ('R13', 'drone tethered, nominal 1.7 m')]
OTHER = [('R07', 'person walking'), ('R08', 'person walking')]

out = {'expected_snr_1m2_1m_db': round(float(base), 1), 'K': {}}
for K in (1.0, 1.5, 1.65, 1.75, 1.9, 2.0):
    rows = []
    for n, lab in DRONE:
        e, r = echo_db(n, K)
        rows.append({'name': n, 'label': lab, 'range_m': round(r, 2), 'echo_db': round(e, 1),
                     'rcs_dbsm': round(float(e - base + 40 * np.log10(r)), 1)})
    drop = rows[0]['echo_db'] - rows[2]['echo_db']
    pred = 40 * np.log10(rows[2]['range_m'] / rows[0]['range_m'])
    rcs = [x['rcs_dbsm'] for x in rows]
    out['K'][str(K)] = {'drone': rows, 'drop_db': round(float(drop), 1), 'r4_drop_db': round(float(pred), 1),
                        'rcs_spread_db': round(float(max(rcs) - min(rcs)), 1)}
    print(f'K={K}: drop {drop:5.1f} dB (R^-4: {pred:.1f}), RCS ' +
          ', '.join(f"{x['range_m']} m {x['rcs_dbsm']} dBsm" for x in rows))
# independent check of K on the test recordings (not used to choose K)
TEST = [('T09', 'pole, room 1'), ('T10', 'pole, room 1'), ('U05', 'sofa arm, room 2'), ('U04', 'sofa arm, room 2')]
for K in (1.5, 1.65, 1.75, 1.9, 2.0):
    rows = []
    for n, lab in TEST:
        e, r = echo_db(n, K, 'data_test')
        rows.append({'name': n, 'label': lab, 'range_m': round(r, 2), 'echo_db': round(e, 1),
                     'rcs_dbsm': round(float(e - base + 40 * np.log10(r)), 1)})
    pairs = {}
    for a_, b_ in (('T09', 'T10'), ('U05', 'U04')):
        x = next(v for v in rows if v['name'] == a_)
        y = next(v for v in rows if v['name'] == b_)
        pairs[f'{a_}->{b_}'] = {'drop_db': round(x['echo_db'] - y['echo_db'], 1),
                                'r4_drop_db': round(float(40 * np.log10(y['range_m'] / x['range_m'])), 1)}
    out.setdefault('test_check', {})[str(K)] = {'recordings': rows, 'pairs': pairs}
    print(f'test check K={K}: ' + ', '.join(f"{k} drop {v['drop_db']} (R^-4 {v['r4_drop_db']})" for k, v in pairs.items())
          + '; RCS ' + ', '.join(f"{x['name']} {x['rcs_dbsm']}" for x in rows))
# the echo also leaks into the neighbouring range rows: how much more with rows -1, 0, +1
out['neighbour_rows_db'] = {}
for n, folder in [(x, 'data') for x, _ in DRONE] + [(x, 'data_test') for x, _ in TEST]:
    one, _ = echo_db(n, 1.75, folder)
    three, _ = echo_db(n, 1.75, folder, rows=(-1, 0, 1))
    out['neighbour_rows_db'][n] = round(three - one, 2)
print('three rows instead of one, K=1.75: ' + ', '.join(f"{k} +{v} dB" for k, v in out['neighbour_rows_db'].items()))

# control: the same estimator on empty rooms (training and test), where there is no target
for K in (1.65, 1.9, 2.0):
    rows = []
    for f in sorted(glob.glob('data/*_empty_*.npz')) + sorted(glob.glob('data_test/*_empty_*.npz')):
        e, r = echo_db(f, K)
        rows.append({'name': rid(f), 'row_m': round(r, 2), 'echo_db': round(e, 1)})
    out.setdefault('empty_control', {})[str(K)] = rows
    print(f'empty-room control K={K}: ' + ', '.join(f"{x['name']} {x['echo_db']}" for x in rows))


def person_rcs(name, K=2.0):
    """A walking person changes range: per frame, take the person's row, subtract the noise
    reference, scale by R^4, then average. Returns (mean RCS, median-frame RCS) in dBsm."""
    s = load_session(glob.glob('data/*' + name + '*')[0])
    p = s.params
    nd = p.doppler_bins
    mv = np.abs(np.arange(nd) - nd // 2) > 1
    vals = []
    for rd in s.rd:
        rb = target_bin(rd, p)
        ref = [r for r in range(8, 60) if abs(r - rb) > 3]
        L = 10 ** (K * (rd - np.median(rd)) / 10)
        vals.append((L[rb, mv].sum() - L[ref][:, mv].sum(1).mean()) * (rb * p.range_bin_m) ** 4)
    vals = np.array(vals)
    lin_base = 10 ** (base / 10)
    return (float(10 * np.log10(vals.mean() / lin_base)),
            float(10 * np.log10(max(np.median(vals), 1e-30) / lin_base)))


for n, lab in OTHER:
    m, md = person_rcs(n)
    out.setdefault('person_K2', []).append({'name': n, 'rcs_mean_dbsm': round(m, 1), 'rcs_median_frame_dbsm': round(md, 1)})
    print(f'{lab}: RCS {m:.1f} dBsm (mean over frames), {md:.1f} dBsm (median frame), K=2')
Path('report/rcs_results.json').write_text(json.dumps(out, indent=1))
