"""Figure of the independent test: output labels per test recording. Run from the repo root:
python3 report/make_figure_test.py   (needs report/test_results.json from compute_test.py)"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'savefig.bbox': 'tight',
                     'savefig.dpi': 220})
COL = {'empty': '#8c8c8c', 'drone': '#c0392b', 'human': '#2b6cb0', 'fan': '#6b8e23', 'unsure': '#e0b84c'}
OUTS = ['empty', 'human', 'fan', 'drone', 'unsure']
R = json.load(open('report/test_results.json'))['recordings']

groups = [
    ('Room 1, training height', [('T04', 'empty'), ('T05', 'empty, me behind radar'), ('T06', 'walking (cut at 18 s)'),
                                 ('T07', 'walking'), ('T03', 'fan, 1.05 m'), ('T08', 'drone on pole, fell at 23.5 s'),
                                 ('T09', 'drone on pole, 0.82 m'), ('T10', 'drone on pole, 1.4 m')]),
    ('Room 1, radar 18 cm lower', [('T01', 'empty'), ('T02', 'walking'), ('T03x', 'fan, 0.94 m')]),
    ('Room 2', [('U01', 'empty, me behind radar'), ('U02', 'walking'), ('U03', 'fan, 1.64 m'),
                ('U04', 'drone, 1.64 m'), ('U05', 'drone, 1.29 m'), ('U06', 'drone, free flight'),
                ('U07', 'drone, free flight')]),
    ('Two moving targets', [('T11', 'fan + me behind it'), ('T12', 'fan + me in front'),
                            ('T13', 'drone + me behind it')]),
]


def rec(rid):
    return next(v for k, v in R.items() if k.split('_')[0] == rid)


n = sum(len(g[1]) for g in groups) + len(groups) * 1.0
fig, ax = plt.subplots(figsize=(6.4, 0.2 * n + 0.6))
y, yt, yl = 0.0, [], []
for title, items in groups:
    ax.text(-1, y + 0.1, title, ha='right', va='bottom', fontsize=8, fontweight='bold', transform=ax.transData,
            clip_on=False)
    y += 0.95
    for rid, desc in items:
        d = rec(rid)['labels']
        left = 0
        for o in OUTS:
            w = 100 * d.get(o, 0)
            if w > 0:
                ax.barh(y, w, left=left, color=COL[o], edgecolor='white', lw=0.4, height=0.8)
                if w >= 8:
                    ax.text(left + w / 2, y, f'{w:.0f}', ha='center', va='center', fontsize=6.5,
                            color='white' if o != 'unsure' else 'k')
                left += w
        yt.append(y)
        yl.append(f'{rid}  {desc}')
        y += 1
    y += 0.3
ax.set_yticks(yt, yl, fontsize=7.5)
ax.tick_params(axis='y', length=0)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel('share of frames (%), frozen model, live smoothing')
ax.legend(handles=[Patch(color=COL[o], label=o) for o in OUTS], ncol=5, loc='lower center',
          bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
fig.savefig('report/figs/test.pdf')
