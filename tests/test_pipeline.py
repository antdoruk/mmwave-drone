from pathlib import Path

import numpy as np
import pytest

from mmwd.config import parse_cfg
from mmwd.features import frame_features
from mmwd.sim import Simulator, scene
from mmwd.tlv import LOG2_TO_DB, MAGIC, StreamParser, encode_packet

ROOT = Path(__file__).resolve().parents[1]

CFG = "configs/md_1tx.cfg"


def test_cfg_matches_ti_visualizer_numbers():
    # the TI visualizer wrote these numbers in the header of this file
    p = parse_cfg("configs/ti_default_68xx.cfg")
    assert p.range_res == pytest.approx(0.044, abs=0.001)
    assert p.v_max == pytest.approx(1.0, abs=0.05)
    assert p.v_res == pytest.approx(0.13, abs=0.01)


def test_md_cfg_fits_uart():
    p = parse_cfg(CFG)
    uart_s = p.rd_bytes * 10 / 921600
    assert uart_s < 0.8 * p.frame_period_ms / 1000


def _packet(k, p, rd):
    return encode_packet(k, points=np.array([[0.1, 2.0, 0.0, 3.0]]),
                         range_profile_db=np.full(p.range_bins, 60.0),
                         rd_db_unshifted=rd, snr_db=[20.0], stats=(1, 2, 3, 4, 5, 6))


def test_roundtrip_exact():
    p = parse_cfg(CFG)
    rd = np.random.default_rng(0).uniform(40, 120, (p.range_bins, p.doppler_bins))
    fr, = StreamParser().feed(_packet(7, p, rd))
    assert fr.number == 7 and fr.num_obj == 1
    back = np.fft.ifftshift(fr.rd_db(p.range_bins, p.doppler_bins), axes=1)
    assert np.max(np.abs(back - rd)) <= LOG2_TO_DB / 2 + 1e-9
    assert fr.points[0, 3] == pytest.approx(3.0)
    assert fr.snr_db[0] == pytest.approx(20.0)
    assert fr.stats["active_cpu_load"] == 5


def test_stream_survives_garbage_and_split_reads():
    p = parse_cfg(CFG)
    rd = np.full((p.range_bins, p.doppler_bins), 50.0)
    stream = b"junk" + _packet(1, p, rd) + MAGIC[:5] + b"xx" + _packet(2, p, rd)[:-100]
    stream += _packet(3, p, rd)                      # frame 2 truncated -> must be dropped
    sp = StreamParser()
    got = []
    for i in range(0, len(stream), 997):             # odd chunk size splits magic words
        got += sp.feed(stream[i:i + 997])
    assert [f.number for f in got] == [1, 3]
    assert sp.bad_packets >= 1


@pytest.mark.parametrize("name", ["empty", "drone", "human", "fan"])
def test_simulator_runs(name):
    p = parse_cfg(CFG)
    sim = Simulator(p)
    (_, rd, rp), = list(sim.frames(scene(name), 1))
    assert rd.shape == (p.range_bins, p.doppler_bins)


def test_features_separate_drone_from_human():
    p = parse_cfg(CFG)
    sim = Simulator(p, seed=3)
    f = {}
    for name in ("drone", "human", "empty"):
        (_, rd, _), = list(sim.frames(scene(name, 2.0), 1))
        f[name] = frame_features(np.fft.fftshift(rd, axes=1), p)
    assert f["drone"]["occupancy"] > f["human"]["occupancy"]
    assert f["drone"]["high_v_frac"] > f["human"]["high_v_frac"]
    assert f["empty"]["snr_db"] < f["drone"]["snr_db"]


def test_malformed_payloads_are_dropped_and_stream_continues():
    import struct
    from mmwd.tlv import HEADER, TLV_HDR, TLV_RD_HEATMAP, TLV_SIDE_INFO
    good = encode_packet(7, range_profile_db=np.zeros(64))

    def packet(tlv_type, payload, n_obj=0):
        body = TLV_HDR.pack(tlv_type, len(payload)) + payload
        total = HEADER.size + len(body)
        return HEADER.pack(MAGIC, 0x03050004, total, 0xA6843, 1, 0, n_obj, 1, 0) + body

    bad_odd = packet(TLV_RD_HEATMAP, bytes(7))                 # odd length for uint16 data
    bad_side = packet(TLV_SIDE_INFO, bytes(6), n_obj=2)        # 6 bytes for 2 points (needs 8)
    sp = StreamParser()
    frames = sp.feed(bad_odd + bad_side + good)
    assert [f.number for f in frames] == [7]
    assert sp.bad_packets == 2


def test_next_id_skips_gaps(tmp_path):
    from mmwd.session import next_id
    assert next_id(tmp_path, "T") == "T01"
    for n in ("T01_empty_0m_0deg.npz", "T03_fan_1m_0deg.npz", "T03_fan_1m_0deg.bin"):
        (tmp_path / n).touch()
    assert next_id(tmp_path, "T") == "T04"
    assert next_id(tmp_path, "U") == "U01"


def test_load_session_rejects_broken_recordings(tmp_path):
    """NaN maps, a wrong map size and a truncated time axis must fail loudly."""
    import glob
    from mmwd.session import load_session
    src = sorted(glob.glob(str(ROOT / "data_test" / "T09_*.npz")))[0]
    z = dict(np.load(src, allow_pickle=False))
    load_session(src)                                   # the original is fine
    for name, change in [("nan", lambda d: d["rd"].__setitem__((0, 0, 0), np.nan)),
                         ("shape", lambda d: d.__setitem__("rd", d["rd"][:, :, :127])),
                         ("time", lambda d: d.__setitem__("t", d["t"][:-1]))]:
        d = {k: v.copy() for k, v in z.items()}
        change(d)
        np.savez(tmp_path / f"{name}.npz", **d)
        with pytest.raises(ValueError):
            load_session(tmp_path / f"{name}.npz")
