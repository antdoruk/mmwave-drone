"""Turn one range-Doppler map into a handful of numbers a human can reason about.

Idea in one sentence: a spinning propeller puts energy into (almost) every
Doppler bin of its range bin, a walking person puts it near +-few m/s, a fan in
between, an empty room nowhere (static clutter is removed by the firmware).

All features are computed on ONE frame, then aggregated over a short window
of frames (default 4 frames = 1 s) for classification.
"""
from __future__ import annotations

import numpy as np

from .config import RadarParams

FEATURES = (
    "snr_db",            # strongest moving return above the map's noise floor
    "occupancy",         # fraction of Doppler bins > floor + 6 dB at the target range
    "high_v_frac",       # share of moving energy at |v| > 10 m/s
    "flatness",          # spectral flatness (1 = white, 0 = single line)
    "v_centroid",        # energy-weighted mean |v|, m/s
    "range_extent",      # number of range bins with strong moving energy
    "period_score",      # how comb-like the Doppler spectrum is (HERM lines)
    "line_spacing_hz",   # spacing of those lines
    "skirt_db",          # mean level 10-25 m/s above the floor: only spinning blades reach it
)


def noise_floor(rd_db: np.ndarray) -> float:
    return float(np.median(rd_db))


# Bins closer than this are ignored: the board shows fixed spurs at 0.1-0.35 m with energy
# near -21 to -24 m/s (seen in every empty-room recording), which would look like blade energy.
MIN_RANGE_M = 0.45

# Doppler bins either side of zero that are treated as "not moving". 1 bin = 0.56 m/s.
# Was 2 (|v| < 1.1 m/s ignored); 1 keeps slow body motion (standing, small hand moves)
# and the empty room still stays below the gate (checked on 5 empty recordings).
GUARD_BINS = 1


def first_bin(params: RadarParams, min_range_m: float = MIN_RANGE_M) -> int:
    return int(np.ceil(min_range_m / params.range_bin_m))


def target_bin(rd_db: np.ndarray, params: RadarParams, min_range_m: float = MIN_RANGE_M,
               guard_bins: int = GUARD_BINS) -> int:
    """Range bin with the most energy away from zero Doppler, beyond the spur zone."""
    nd = rd_db.shape[1]
    lin = 10 ** (rd_db / 10)
    mask = np.ones(nd, bool)
    mask[nd // 2 - guard_bins: nd // 2 + guard_bins + 1] = False
    e = lin[:, mask].sum(1)
    e[: max(1, first_bin(params, min_range_m))] = 0
    return int(np.argmax(e))


def doppler_spectrum(rd_db: np.ndarray, rb: int, width: int = 1) -> np.ndarray:
    lo, hi = max(0, rb - width), min(rd_db.shape[0], rb + width + 1)
    return rd_db[lo:hi].max(0)


def comb_period(spec_db: np.ndarray, min_lag: int = 2, max_frac: float = 0.25):
    """Autocorrelation of the detrended spectrum. Returns (score, lag_bins).
    A comb of equally spaced lines gives a clear autocorrelation peak at the spacing."""
    x = spec_db - np.convolve(spec_db, np.ones(9) / 9, mode="same")
    x = x - x.mean()
    n = len(x)
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] <= 0:
        return 0.0, 0
    ac = ac / ac[0]
    hi = max(min_lag + 2, int(n * max_frac))
    seg = ac[min_lag:hi]
    k = int(np.argmax(seg))
    return float(seg[k]), k + min_lag


def frame_features(rd_db: np.ndarray, params: RadarParams) -> dict:
    """rd_db: (range_bins, doppler_bins), zero Doppler in the middle (fftshifted)."""
    nd = rd_db.shape[1]
    floor = noise_floor(rd_db)
    rb = target_bin(rd_db, params)
    spec = doppler_spectrum(rd_db, rb)
    v = params.velocity_axis()
    moving = np.abs(np.arange(nd) - nd // 2) > GUARD_BINS
    lin = 10 ** ((spec - floor) / 10)
    lin_m = np.where(moving, lin, 0)
    tot = lin_m.sum() + 1e-12
    above = spec[moving] > floor + 6
    gm = np.exp(np.mean(np.log(lin[moving] + 1e-12)))
    flat = gm / (lin[moving].mean() + 1e-12)
    strong_rows = (rd_db[first_bin(params):, moving].max(1) > floor + 15).sum()
    score, lag = comb_period(spec)
    band = (np.abs(v) > 10) & (np.abs(v) < 25)
    skirt = 10 * np.log10(np.mean(10 ** ((spec[band] - floor) / 10)))
    return {
        "snr_db": float(spec[moving].max() - floor),
        "occupancy": float(above.mean()),
        "high_v_frac": float(lin_m[np.abs(v) > 10].sum() / tot),
        "flatness": float(flat),
        "v_centroid": float((np.abs(v) * lin_m).sum() / tot),
        "range_extent": float(strong_rows),
        "period_score": score,
        "line_spacing_hz": float(lag * params.doppler_bin_hz),
        "skirt_db": float(skirt),
        "_range_m": float(rb * params.range_bin_m),
        "_floor_db": floor,
    }


def window_features(frames_feats: list[dict]) -> np.ndarray:
    """Mean and std over a window of per-frame feature dicts -> 1D vector."""
    a = np.array([[f[k] for k in FEATURES] for f in frames_feats], dtype=float)
    return np.concatenate([a.mean(0), a.std(0)])


def window_feature_names() -> list[str]:
    return [f"{k}_mean" for k in FEATURES] + [f"{k}_std" for k in FEATURES]
