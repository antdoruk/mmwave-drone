"""One recording = one .npz (parsed frames) + one .bin (raw bytes from the board)
+ metadata (label, distance, angle, notes, the exact cfg). Raw bytes are kept so
that any future parser fix can re-process old measurements."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RadarParams, parse_cfg
from .tlv import Frame, StreamParser


@dataclass
class Session:
    meta: dict
    params: RadarParams
    rd: np.ndarray            # (frames, range, doppler) dB, zero Doppler centred
    range_profile: np.ndarray # (frames, range) dB (may be empty)
    t: np.ndarray             # (frames,) host time, s
    frame_no: np.ndarray      # (frames,)
    points: list = field(default_factory=list)  # list of (N,4) arrays

    @property
    def label(self) -> str:
        return self.meta.get("label", "?")

    @property
    def name(self) -> str:
        return self.meta.get("name", "?")


def frames_to_arrays(frames: list[Frame], params: RadarParams):
    """Stack frames into arrays. Normal recordings keep only frames with a range-Doppler
    map. A profile-only recording (RD heatmap switched off in the cfg, e.g. the fast
    breathing profile) keeps every frame with a range profile and an empty rd array."""
    profile_only = not any(f.rd_db(params.range_bins, params.doppler_bins) is not None for f in frames)
    rds, rps, ts, nos, pts = [], [], [], [], []
    for f in frames:
        rd = f.rd_db(params.range_bins, params.doppler_bins)
        if rd is None and not (profile_only and f.range_profile is not None):
            continue
        if rd is not None:
            rds.append(rd)
        rps.append(f.range_profile if f.range_profile is not None else np.zeros(params.range_bins))
        ts.append(f.timestamp)
        nos.append(f.number)
        pts.append(f.points if f.points is not None else np.zeros((0, 4), np.float32))
    if not rps:
        return (np.zeros((0, params.range_bins, params.doppler_bins), np.float32),
                np.zeros((0, params.range_bins)), np.zeros(0), np.zeros(0, int), [])
    rd = (np.stack(rds).astype(np.float32) if rds
          else np.zeros((0, params.range_bins, params.doppler_bins), np.float32))
    return (rd, np.stack(rps).astype(np.float32), np.array(ts), np.array(nos), pts)


def save_session(path_stem: str | Path, frames: list[Frame], params: RadarParams, cfg_text: str,
                 meta: dict) -> Path:
    path_stem = Path(path_stem)
    rd, rp, t, no, pts = frames_to_arrays(frames, params)
    meta = dict(meta)
    meta.update(name=path_stem.name,
                n_frames=int(len(t)), cfg=cfg_text, params=params.to_dict())
    lens = np.array([len(p) for p in pts], int)
    flat = np.concatenate(pts) if pts else np.zeros((0, 4), np.float32)
    # times relative to the first frame: the file keeps durations, not the wall-clock time
    t = t - t[0] if len(t) else t
    np.savez_compressed(path_stem.with_suffix(".npz"), rd=rd, range_profile=rp, t=t, frame_no=no,
                        points=flat, points_len=lens, meta=json.dumps(meta))
    return path_stem.with_suffix(".npz")


def load_session(path: str | Path) -> Session:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    params = RadarParams(**meta["params"])
    lens = z["points_len"]
    flat = z["points"]
    pts = np.split(flat, np.cumsum(lens)[:-1]) if len(lens) else []
    rd, t, no = z["rd"], z["t"], z["frame_no"]
    _check(path, params, rd, t, no)
    return Session(meta=meta, params=params, rd=rd, range_profile=z["range_profile"],
                   t=t, frame_no=no, points=pts)


def _check(path, params: RadarParams, rd, t, no):
    """Reject recordings whose arrays do not fit together, instead of failing later or,
    worse, silently giving labels. A recording without range-Doppler maps (profile only,
    or cut before the first frame) is allowed; callers decide what to do with it."""
    name = Path(path).name
    if rd.ndim != 3 or (len(rd) and rd.shape[1:] != (params.range_bins, params.doppler_bins)):
        raise ValueError(f"{name}: range-Doppler maps have shape {rd.shape}, expected "
                         f"(n, {params.range_bins}, {params.doppler_bins})")
    if len(t) != len(no) or (len(rd) and len(rd) != len(t)):
        raise ValueError(f"{name}: {len(rd)} maps, {len(t)} times, {len(no)} frame numbers")
    if len(rd) and not np.isfinite(rd).all():
        raise ValueError(f"{name}: range-Doppler maps contain NaN or infinite values")
    if len(t) > 1 and np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: frame times are not increasing")


def reparse_raw(bin_path: str | Path, cfg_path: str | Path) -> tuple[list[Frame], RadarParams]:
    params = parse_cfg(cfg_path)
    frames = StreamParser().feed(Path(bin_path).read_bytes())
    return frames, params


def next_id(folder, prefix: str) -> str:
    """Next recording ID: one more than the highest number already in `folder`, so a gap
    (T01, T03) gives T04 and never reuses a name."""
    nums = [int(m.group(1)) for f in Path(folder).glob(f"{prefix}[0-9]*")
            if (m := re.match(re.escape(prefix) + r"(\d+)", f.name))]
    return f"{prefix}{max(nums, default=0) + 1:02d}"
