"""Live view: range-Doppler map, Doppler spectrum at the target, label.

    python3 scripts/live.py                        # real board
    python3 scripts/live.py --sim drone            # simulator (also: human, fan, empty)
    python3 scripts/live.py --model models/frozen/model_frozen.joblib

Keys in the window: s = save a screenshot to figures/, q = quit.
Without --model a simple hand-written rule is shown instead of the classifier.
"""
import argparse
import time
from pathlib import Path

import _path  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from mmwd.cli import add_source_args, open_source
from mmwd.features import frame_features

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_source_args(ap)
ap.add_argument("--model", help="trained model from scripts/train.py")
ap.add_argument("--max-frames", type=int, default=0, help=argparse.SUPPRESS)
ap.add_argument("--dyn", type=float, default=40.0, help="colour range of the map in dB")
args = ap.parse_args()

src, p, cli = open_source(args)
clf = None
if args.model:
    from mmwd.classify import LiveClassifier
    clf = LiveClassifier(args.model, p)


def rule(f):
    """Hand-written baseline, before any training."""
    if f["snr_db"] < 12:
        return "empty", None
    # thresholds from simulated frames (drone: occupancy >= 0.6, human/fan <= 0.25);
    # a rough baseline only, the trained model (--model) is what the report evaluates
    if f["occupancy"] > 0.4 and f["high_v_frac"] > 0.15:
        return "drone?", None
    return "moving (not drone?)", None


v = p.velocity_axis()
r = p.range_axis()
plt.ion()
fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
ax_rd = fig.add_subplot(gs[:, 0])
ax_sp = fig.add_subplot(gs[0, 1])
ax_rp = fig.add_subplot(gs[1, 1])
ax_tx = fig.add_subplot(gs[2, 1])
ax_tx.axis("off")
im = ax_rd.imshow(np.zeros((p.range_bins, p.doppler_bins)), aspect="auto", origin="lower",
                  extent=[v[0], v[-1], 0, r[-1] + p.range_bin_m], cmap="viridis")
ax_rd.set_xlabel("radial velocity (m/s)")
ax_rd.set_ylabel("range (m)")
ax_rd.set_title("range-Doppler map (static objects removed)")
mark = ax_rd.axhline(0, color="w", lw=0.8, ls="--")
(l_sp,) = ax_sp.plot(v, np.zeros_like(v), lw=1)
(l_fl,) = ax_sp.plot(v, np.zeros_like(v), "k--", lw=0.8)
ax_sp.set_xlabel("radial velocity (m/s)")
ax_sp.set_ylabel("dB")
ax_sp.set_title("Doppler spectrum at the target range")
(l_rp,) = ax_rp.plot(r, np.zeros_like(r), lw=1)
ax_rp.set_xlabel("range (m)")
ax_rp.set_ylabel("dB")
ax_rp.set_title("range profile (incl. static objects)")
txt_big = ax_tx.text(0.0, 0.75, "", fontsize=26, weight="bold", transform=ax_tx.transAxes)
txt = ax_tx.text(0.0, 0.0, "", fontsize=9, family="monospace", transform=ax_tx.transAxes)
fig.tight_layout()

state = {"quit": False, "shot": False}


def on_key(e):
    if e.key == "q":
        state["quit"] = True
    if e.key == "s":
        state["shot"] = True


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("close_event", lambda e: state.update(quit=True))

src.start()
n, t_last, fps = 0, time.time(), 0.0
try:
    while not state["quit"]:
        f = src.get(timeout=0.5)
        if getattr(src, "lost", False):
            break
        if f is None:
            plt.pause(0.01)
            continue
        rd = f.rd_db(p.range_bins, p.doppler_bins)
        if rd is None:
            continue
        floor = np.median(rd)
        im.set_data(rd)
        im.set_clim(floor, floor + args.dyn)
        if clf:
            label, prob, feat = clf.update(rd)
            label = label or "..."
        else:
            feat = frame_features(rd, p)
            label, prob = rule(feat)
        rb = int(round(feat["_range_m"] / p.range_bin_m))
        mark.set_ydata([feat["_range_m"]] * 2)
        spec = rd[rb]
        l_sp.set_ydata(spec)
        l_fl.set_ydata(np.full_like(v, floor + 6))
        ax_sp.set_ylim(floor - 5, max(floor + 20, spec.max() + 5))
        if f.range_profile is not None and len(f.range_profile) == len(r):
            l_rp.set_ydata(f.range_profile)
            ax_rp.set_ylim(f.range_profile.min() - 3, f.range_profile.max() + 3)
        n += 1
        now = time.time()
        fps = 0.8 * fps + 0.2 / max(now - t_last, 1e-3)
        t_last = now
        txt_big.set_text(label + (f"  {prob:.0%}" if prob is not None else ""))
        txt.set_text("\n".join([
            f"frame {f.number}   {fps:4.1f} fps   target at {feat['_range_m']:.2f} m",
            f"SNR {feat['snr_db']:5.1f} dB   occupancy {feat['occupancy']:.2f}   "
            f"|v|>10 m/s share {feat['high_v_frac']:.2f}",
            f"flatness {feat['flatness']:.2f}   mean |v| {feat['v_centroid']:.1f} m/s",
            f"comb score {feat['period_score']:.2f}   line spacing {feat['line_spacing_hz']:.0f} Hz",
        ]))
        if state["shot"]:
            Path("figures").mkdir(exist_ok=True)
            out = Path("figures") / time.strftime("live_%Y%m%d_%H%M%S.png")
            fig.savefig(out, dpi=150)
            print("saved", out)
            state["shot"] = False
        fig.canvas.draw_idle()
        plt.pause(0.001)
        if args.max_frames and n >= args.max_frames:
            fig.savefig("figures/live_test.png", dpi=90)
            break
finally:
    src.stop()
    if cli:
        from mmwd.radar import stop_sensor
        stop_sensor(cli)
