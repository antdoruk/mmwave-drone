"""Physics simulator: point scatterers -> FMCW IF samples -> the same processing
chain as the TI demo -> range-Doppler heatmap (and a full UART packet).

Why it exists
  1. Lets every script run without the board (tests, development, demos).
  2. A physics check: compare the measured propeller signature with the model.
     If the simulated and measured Doppler spectra agree, we understand what
     we measured; if they do not, that disagreement is itself a result.

Model
  * Radar at the origin, boresight +y, x to the right, z up (TI convention).
  * IWR6843AOP receive antennas: 2 x 2 grid, half-wavelength spacing.
  * Each target is a set of point scatterers whose positions are evaluated at
    each chirp time (stop-and-hop: the target does not move during one chirp).
    This is NOT negligible for blade tips: at 50 m/s the tip moves 0.64 mm during
    the 12.8 us ADC window, a round-trip phase change of about 1.6 rad. In a real
    radar that shifts the tip's echo by about 3 cm in range (a quarter of a range
    bin); the simulator ignores it, so blade spectra are approximate.
  * IF sample n of chirp m on antenna k:
        a * exp(j*2*pi*(2*S*R/c * t_n + 2*R/lambda + phase_k))
    with amplitude a ~ sqrt(rcs) / R^2 (two-way radar equation).
  * Processing: Hann window, range FFT, optional static clutter removal
    (subtract the mean over chirps), Hann window, Doppler FFT, magnitude summed
    over antennas, 20*log10. The firmware's internal scaling is unknown to us,
    so absolute dB values are not comparable to the board; shapes are.
  * Differences from the board that are known: the board appears to average
    log-magnitudes over antennas (report, Section 7.1), the simulator sums
    magnitudes; the simulated range profile is taken from the first chirp before
    clutter removal, while the board's profile is computed after it. The
    simulator is an approximate model for testing the software, not a firmware twin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import C, RadarParams


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------
class Target:
    def scatterers(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return positions (T, N, 3) and amplitudes (N,) for chirp times t (T,)."""
        raise NotImplementedError


@dataclass
class Static(Target):
    pos: tuple = (0.0, 2.0, 0.0)
    rcs: float = 0.05

    def scatterers(self, t):
        p = np.broadcast_to(np.asarray(self.pos, float), (len(t), 1, 3))
        return p, np.array([np.sqrt(self.rcs)])


@dataclass
class Rotor:
    center: tuple            # offset from the drone hub, metres
    radius: float = 0.023    # blade tip radius
    blades: int = 2
    rps: float = 500.0       # revolutions per second
    direction: int = 1
    phase0: float = 0.0
    points: int = 8          # scatterers per blade
    rcs_blade: float = 2e-4  # whole blade, m^2 (thin plastic, tiny)


@dataclass
class Drone(Target):
    """Quadcopter (default: palm-sized toy, ~2.3 cm blades, ~30 000 rpm)."""
    hub: tuple = (0.0, 2.0, 0.0)
    velocity: tuple = (0.0, 0.0, 0.0)
    arm: float = 0.045
    radius: float = 0.023
    blades: int = 2
    rps: float = 500.0
    rps_spread: float = 0.04     # rotors do not spin at exactly the same speed
    rcs_body: float = 2e-3
    rcs_blade: float = 2e-4
    seed: int = 0
    rotors: list = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if not self.rotors:
            offs = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
            self.rotors = [
                Rotor(center=(sx * self.arm, sy * self.arm, 0.01), radius=self.radius,
                      blades=self.blades,
                      rps=self.rps * (1 + self.rps_spread * rng.standard_normal()),
                      direction=1 if i % 2 == 0 else -1,
                      phase0=rng.uniform(0, 2 * np.pi), rcs_blade=self.rcs_blade)
                for i, (sx, sy) in enumerate(offs)
            ]

    def scatterers(self, t):
        hub = np.asarray(self.hub, float)[None, :] + np.outer(t, self.velocity)   # (T,3)
        pos = [hub[:, None, :]]
        amp = [np.sqrt(self.rcs_body)]
        for r in self.rotors:
            rho = np.linspace(0.25, 1.0, r.points) * r.radius                      # (P,)
            a_pt = np.sqrt(r.rcs_blade) / r.points
            for b in range(r.blades):
                ang = r.direction * 2 * np.pi * r.rps * t + r.phase0 + 2 * np.pi * b / r.blades
                d = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1)   # (T,3)
                p = hub[:, None, :] + np.asarray(r.center)[None, None, :] + d[:, None, :] * rho[None, :, None]
                pos.append(p)
                amp.extend([a_pt] * r.points)
        return np.concatenate(pos, axis=1), np.asarray(amp)


@dataclass
class Fan(Target):
    """Desk fan seen face-on (blades rotate in the x-z plane, facing the radar)
    or edge-on. Bigger, slower blades than a drone: a useful 'hard negative'."""
    center: tuple = (0.0, 1.5, 0.0)
    radius: float = 0.18
    blades: int = 5
    rps: float = 20.0
    tilt_deg: float = 20.0       # blade pitch gives a radial velocity component
    rcs_blade: float = 5e-3
    points: int = 10

    def scatterers(self, t):
        c = np.asarray(self.center, float)
        rho = np.linspace(0.2, 1.0, self.points) * self.radius
        pos, amp = [], []
        tilt = np.deg2rad(self.tilt_deg)
        for b in range(self.blades):
            ang = 2 * np.pi * self.rps * t + 2 * np.pi * b / self.blades
            # blade direction in the x-z plane; pitch adds a y (radial) component
            d = np.stack([np.cos(ang), np.sin(tilt) * np.sin(ang), np.cos(tilt) * np.sin(ang)], -1)
            pos.append(c[None, None, :] + d[:, None, :] * rho[None, :, None])
            amp.extend([np.sqrt(self.rcs_blade) / self.points] * self.points)
        pos.append(np.broadcast_to(c, (len(t), 1, 3)))
        amp.append(np.sqrt(0.02))
        return np.concatenate(pos, axis=1), np.asarray(amp)


@dataclass
class Human(Target):
    """Very simple walker: torso, head, two arms, two legs swinging in anti-phase.
    walk_speed > 0 walks back and forth, 0 stands (breathing + sway only)."""
    start: tuple = (0.0, 3.0, 0.0)
    walk_speed: float = 1.2
    heading_deg: float = 0.0     # 0 = radial (straight away), 90 = sideways
    cadence_hz: float = 0.9      # stride frequency
    path_m: float = 1.0          # walks +-path_m around start
    seed: int = 1

    def scatterers(self, t):
        rng = np.random.default_rng(self.seed)
        h = np.deg2rad(self.heading_deg)
        fwd = np.array([np.sin(h), np.cos(h), 0.0])
        # walk back and forth over +-path_m (triangle wave) so long recordings stay in range
        if self.walk_speed > 0.05:
            T = 4 * self.path_m / self.walk_speed
            s = self.path_m * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * t / T))
        else:
            s = np.zeros_like(t)
        base = np.asarray(self.start, float)[None, :] + np.outer(s, fwd)
        w = 2 * np.pi * self.cadence_hz * t
        moving = self.walk_speed > 0.05
        sway = 0.004 * np.sin(2 * np.pi * 0.25 * t)                 # breathing ~4 mm
        parts = []
        # (offset z, swing amplitude along fwd, phase, rcs)
        torso = base + np.outer(sway, fwd) + np.outer(0.02 * np.sin(2 * w) * moving, [0, 0, 1])
        parts.append((torso + [0, 0, 0.3], 0.5))
        parts.append((torso + [0, 0, 0.75], 0.05))
        for side, ph in ((+1, 0.0), (-1, np.pi)):
            arm = torso + [0.2 * side, 0, 0.35] + np.outer(0.25 * np.sin(w + ph) * moving, fwd)
            leg = torso + [0.1 * side, 0, -0.45] + np.outer(0.35 * np.sin(w + ph + np.pi) * moving, fwd)
            parts.append((arm, 0.03))
            parts.append((leg, 0.08))
        pos = np.stack([p for p, _ in parts], axis=1)
        amp = np.sqrt(np.array([a for _, a in parts])) * (1 + 0.05 * rng.standard_normal(len(parts)))
        return pos, amp


# ---------------------------------------------------------------------------
# radar
# ---------------------------------------------------------------------------
AOP_RX = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])   # (x, z) in wavelengths


@dataclass
class Simulator:
    params: RadarParams
    noise_db: float = -42.0      # noise per IF sample relative to a 1 m^2 target at 1 m
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def frame_cube(self, targets: list[Target], t0: float) -> np.ndarray:
        """Complex IF cube (rx, chirps, samples) for one frame starting at t0."""
        p = self.params
        lam = p.wavelength
        S = p.slope_mhz_us * 1e12
        fs = p.sample_rate_ksps * 1e3
        tn = p.adc_start_us * 1e-6 + np.arange(p.num_adc) / fs                   # (Ns,)
        tm = t0 + np.arange(p.chirps_per_frame) * p.tc_eff                        # (Nc,)
        cube = np.zeros((p.num_rx, p.chirps_per_frame, p.num_adc), complex)
        for tgt in targets:
            pos, amp = tgt.scatterers(tm)                                         # (Nc,N,3),(N,)
            R = np.linalg.norm(pos, axis=-1)                                      # (Nc,N)
            ux = pos[..., 0] / R
            uz = pos[..., 2] / R
            a = amp[None, :] / R ** 2
            beat = 2 * S * R / C                                                  # Hz
            ph0 = 4 * np.pi * R / lam
            for k, (ax, az) in enumerate(AOP_RX):
                ph_ant = 2 * np.pi * (ax * ux + az * uz)                         # (Nc,N)
                # sum over scatterers: (Nc,N) x (N,Ns) via einsum on exponentials
                e = np.exp(1j * (ph0 + ph_ant))[..., None] * \
                    np.exp(1j * 2 * np.pi * beat[..., None] * tn[None, None, :])
                cube[k] += np.einsum("cn,cns->cs", a, e)
        sigma = 10 ** (self.noise_db / 20)
        cube += sigma / np.sqrt(2) * (self.rng.standard_normal(cube.shape)
                                      + 1j * self.rng.standard_normal(cube.shape))
        return cube

    def process(self, cube: np.ndarray, clutter_removal: bool | None = None):
        """Range and Doppler FFTs as in the demo. Returns (rd_db_unshifted (range, doppler),
        range_profile_db). Unlike the demo, which sends the zero-Doppler column, the range
        profile here is taken from the first chirp before clutter removal."""
        p = self.params
        cr = p.clutter_removal if clutter_removal is None else clutter_removal
        wr = np.hanning(p.num_adc)
        rng_fft = np.fft.fft(cube * wr, n=p.range_bins, axis=-1)                 # (rx,Nc,Nr)
        range_profile = 20 * np.log10(np.abs(rng_fft[:, 0, :]).sum(0) + 1e-12)
        if cr:
            rng_fft = rng_fft - rng_fft.mean(axis=1, keepdims=True)
        wd = np.hanning(p.chirps_per_frame)[None, :, None]
        dop = np.fft.fft(rng_fft * wd, n=p.doppler_bins, axis=1)                 # (rx,Nd,Nr)
        mag = np.abs(dop).sum(0).T                                                # (Nr,Nd)
        rd_db = 20 * np.log10(mag + 1e-12)
        return rd_db, range_profile

    def frames(self, targets: list[Target], n_frames: int, t_start: float = 0.0,
               offset_db: float = 150.0):
        """Yield (t0, rd_db_unshifted, range_profile_db) for consecutive frames.
        offset_db shifts everything positive so it fits the firmware's unsigned Q9 format."""
        for i in range(n_frames):
            t0 = t_start + i * p_frame(self.params)
            cube = self.frame_cube(targets, t0)
            rd, rp = self.process(cube)
            yield t0, rd + offset_db, rp + offset_db


def p_frame(p: RadarParams) -> float:
    return p.frame_period_ms * 1e-3


def detections_from_rd(rd_db_unshifted: np.ndarray, params: RadarParams, thresh_db: float = 15.0,
                       max_points: int = 20) -> np.ndarray:
    """Crude stand-in for CFAR: strongest cells above median+thresh -> (x,y,z,v) points
    on boresight. Only used to fill the point-cloud TLV in simulated packets."""
    rd = np.fft.fftshift(rd_db_unshifted, axes=1)
    floor = np.median(rd)
    idx = np.argwhere(rd > floor + thresh_db)
    if len(idx) == 0:
        return np.zeros((0, 4), np.float32)
    vals = rd[idx[:, 0], idx[:, 1]]
    idx = idx[np.argsort(-vals)][:max_points]
    r = idx[:, 0] * params.range_bin_m
    v = params.velocity_axis()[idx[:, 1]]
    return np.stack([np.zeros_like(r), r, np.zeros_like(r), v], 1).astype(np.float32)


def scene(name: str, distance: float = 2.0, seed: int = 0) -> list[Target]:
    """Named scenes used by tests, the fake board and the simulation study."""
    if name == "empty":
        return [Static((0.4, 4.5, 0.0), 0.5), Static((-1.0, 6.0, 0.3), 1.0)]
    if name == "drone":
        return [Drone(hub=(0.0, distance, 0.0), seed=seed),
                Static((0.0, distance + 0.6, 0.0), 0.01),              # the stick
                Static((0.4, 4.5, 0.0), 0.5)]
    if name == "drone_stick_hand":
        return scene("drone", distance, seed) + [Human(start=(0.0, distance + 1.0, 0.0), walk_speed=0.0, seed=seed)]
    if name == "human":
        return [Human(start=(0.0, distance, 0.0), walk_speed=1.0, seed=seed),
                Static((0.4, 4.5, 0.0), 0.5)]
    if name == "human_still":
        return [Human(start=(0.0, distance, 0.0), walk_speed=0.0, seed=seed)]
    if name == "fan":
        return [Fan(center=(0.0, distance, 0.0)), Static((0.4, 4.5, 0.0), 0.5)]
    raise ValueError(name)


SCENES = ("empty", "drone", "human", "fan")
