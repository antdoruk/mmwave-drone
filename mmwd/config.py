"""Read a TI mmWave SDK 3.x .cfg file and derive the radar parameters we need.

Everything downstream (axes of the range-Doppler map, simulator, feature
extraction) uses RadarParams, so the numbers on every plot come from the same
place as the chirp that was actually sent to the board.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

C = 299_792_458.0


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@dataclass
class RadarParams:
    start_freq_ghz: float
    idle_us: float
    adc_start_us: float
    ramp_end_us: float
    slope_mhz_us: float
    num_adc: int
    sample_rate_ksps: float
    num_tx: int
    num_rx: int
    chirps_per_frame: int      # chirps per TX per frame (numLoops)
    frame_period_ms: float
    clutter_removal: bool

    # ---- derived quantities -------------------------------------------------
    @property
    def wavelength(self) -> float:
        fc = (self.start_freq_ghz * 1e9
              + self.slope_mhz_us * 1e12 * (self.adc_start_us * 1e-6 + 0.5 * self.adc_time))
        return C / fc

    @property
    def adc_time(self) -> float:
        return self.num_adc / (self.sample_rate_ksps * 1e3)

    @property
    def bandwidth(self) -> float:
        """Bandwidth actually seen by the ADC (sets range resolution)."""
        return self.slope_mhz_us * 1e12 * self.adc_time

    @property
    def range_bins(self) -> int:
        return _next_pow2(self.num_adc)

    @property
    def range_res(self) -> float:
        return C / (2 * self.bandwidth)

    @property
    def range_bin_m(self) -> float:
        """Spacing of FFT bins (equals range_res when num_adc is a power of two)."""
        return self.range_res * self.num_adc / self.range_bins

    @property
    def max_range(self) -> float:
        # complex baseband: IF bandwidth = sample rate
        return self.sample_rate_ksps * 1e3 * C / (2 * self.slope_mhz_us * 1e12)

    @property
    def chirp_time(self) -> float:
        return (self.idle_us + self.ramp_end_us) * 1e-6

    @property
    def tc_eff(self) -> float:
        """Time between two chirps of the same TX (Doppler sampling period)."""
        return self.chirp_time * self.num_tx

    @property
    def doppler_bins(self) -> int:
        return _next_pow2(self.chirps_per_frame)

    @property
    def prf(self) -> float:
        return 1.0 / self.tc_eff

    @property
    def v_max(self) -> float:
        return self.wavelength / (4 * self.tc_eff)

    @property
    def doppler_bin_hz(self) -> float:
        return self.prf / self.doppler_bins

    @property
    def v_res(self) -> float:
        return self.wavelength / (2 * self.doppler_bins * self.tc_eff)

    @property
    def dwell(self) -> float:
        return self.chirps_per_frame * self.tc_eff

    @property
    def fps(self) -> float:
        return 1000.0 / self.frame_period_ms

    @property
    def rd_bytes(self) -> int:
        return self.range_bins * self.doppler_bins * 2

    def range_axis(self):
        import numpy as np
        return np.arange(self.range_bins) * self.range_bin_m

    def doppler_axis_hz(self):
        """Doppler frequency of each column AFTER fftshift (negative to positive)."""
        import numpy as np
        n = self.doppler_bins
        return (np.arange(n) - n // 2) * self.doppler_bin_hz

    def velocity_axis(self):
        return self.doppler_axis_hz() * self.wavelength / 2

    def summary(self) -> str:
        lines = [
            f"wavelength           {self.wavelength*1e3:.2f} mm",
            f"ADC bandwidth        {self.bandwidth/1e9:.2f} GHz",
            f"range resolution     {self.range_res*100:.1f} cm",
            f"max range            {self.max_range:.2f} m",
            f"range bins           {self.range_bins}",
            f"chirp period (1 TX)  {self.tc_eff*1e6:.1f} us",
            f"max |velocity|       {self.v_max:.1f} m/s",
            f"Doppler bins         {self.doppler_bins}",
            f"Doppler bin          {self.doppler_bin_hz:.0f} Hz  ({self.v_res:.2f} m/s)",
            f"dwell per frame      {self.dwell*1e3:.2f} ms",
            f"frame rate           {self.fps:.1f} fps",
            f"RD heatmap per frame {self.rd_bytes/1024:.1f} KB",
            f"UART load (921600)   {self.rd_bytes*10/921600*1000:.0f} ms of {self.frame_period_ms:.0f} ms frame",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def read_cfg_lines(path: str | Path) -> list[str]:
    out = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        out.append(line)
    return out


def parse_cfg(path_or_lines) -> RadarParams:
    lines = read_cfg_lines(path_or_lines) if isinstance(path_or_lines, (str, Path)) else path_or_lines
    cmd = {}
    chirps = []
    for line in lines:
        parts = line.split()
        if parts[0] == "chirpCfg":
            chirps.append([float(x) for x in parts[1:]])
        elif parts[0] == "cfarCfg" or parts[0] == "cfarFovCfg":
            continue
        else:
            cmd[parts[0]] = [float(x) if _is_num(x) else x for x in parts[1:]]

    p = cmd["profileCfg"]
    ch = cmd["channelCfg"]
    fr = cmd["frameCfg"]
    rx_mask, tx_mask = int(ch[0]), int(ch[1])
    num_rx = bin(rx_mask).count("1")
    num_tx = int(fr[1] - fr[0] + 1)          # chirps in one loop = TXs in TDM
    tx_bits = bin(tx_mask).count("1")
    if num_tx != tx_bits:
        # not an error for the firmware, but worth knowing
        pass
    clutter = bool(int(cmd.get("clutterRemoval", [-1, 0])[1]))
    return RadarParams(
        start_freq_ghz=p[1], idle_us=p[2], adc_start_us=p[3], ramp_end_us=p[4],
        slope_mhz_us=p[7], num_adc=int(p[9]), sample_rate_ksps=p[10],
        num_tx=num_tx, num_rx=num_rx, chirps_per_frame=int(fr[2]),
        frame_period_ms=fr[4], clutter_removal=clutter,
    )


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print(f"== {f}")
        print(parse_cfg(f).summary())
