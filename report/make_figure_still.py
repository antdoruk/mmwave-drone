"""Figure of Section 9 (a person standing still, clutter removal off, four frames per second).

    python3 report/make_figure_still.py      -> report/figs/fig_breath_test.png

Top: change of the range profile in a bin with only static objects and in the bin where the
person stood. Bottom: standard deviation at the person's bin over a sliding 10 s window.
The breath-hold interval is taken from the recording note.
"""
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmwd.session import load_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
s = load_session(glob.glob(str(ROOT / "data" / "*B02*.npz"))[0])   # note: 0-40 s deep breathing, 40-55 s held
HOLD = (40.0, 55.0)
rb = s.params.range_bin_m
t, rp = s.t - s.t[0], s.range_profile
static_b, person_b = int(round(0.82 / rb)), int(round(1.17 / rb))
x_static = rp[:, static_b] - rp[:, static_b].mean()
x_person = rp[:, person_b] - rp[:, person_b].mean()
w = int(round(10 * s.params.fps))
sd = np.array([x_person[i:i + w].std() for i in range(len(t) - w + 1)])
tc = t[:len(sd)] + 0.5 * (t[w - 1] - t[0])                 # centre of each window
calm = tc[sd.argmin()]

fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
for a in ax:
    a.axvspan(*HOLD, color="tab:green", alpha=0.12, lw=0)
ax[0].plot(t, x_static + 10, color="grey", label=f"{static_b * rb:.2f} m: walls and furniture only (offset +10 dB)")
ax[0].plot(t, x_person, color="tab:red", label=f"{person_b * rb:.2f} m: person standing still")
ax[0].set_ylabel("change (dB)")
ax[0].legend(loc="upper right", fontsize=8)
ax[0].text(np.mean(HOLD), ax[0].get_ylim()[0] + 0.5, "breath held", color="green", ha="center", fontsize=9)
ax[1].plot(tc, sd, color="tab:blue")
ax[1].axvline(calm, color="k", ls=":", lw=1)
ax[1].text(calm, sd.max(), f" calmest 10 s window\n (centred at {calm:.0f} s)", fontsize=8, va="top")
ax[1].set_ylabel("std at the person, 10 s window (dB)")
ax[1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(ROOT / "report" / "figs" / "fig_breath_test.png", dpi=150)
print(f"static bin std {x_static.std():.2f} dB, person bin std {x_person.std():.2f} dB, calmest window at {calm:.1f} s")
