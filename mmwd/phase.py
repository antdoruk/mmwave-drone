"""Phase retrieval on the board's magnitude-only range profile.

Why this is a phase retrieval problem
  With static clutter removal OFF, a range cell holds the echo of a slowly moving
  target (chest, pendulum) plus the echoes of everything static in that cell.
  For antenna k and range cell b:
        X_kb(t) = A_kb * exp(j*phi(t)) + C_kb,      phi(t) = 4*pi*d(t)/lambda
  d(t) is the target's displacement; one full turn of phi is lambda/2 = 2.45 mm.
  The board sends no phase, only one magnitude per cell combined over the 4 antennas
  (according to TI, a sum of log2 magnitudes in Q9). As a model, take the summed power:
        y_b(t) = sum_k |X_kb(t)|^2 = alpha_b + Re(gamma_b * exp(j*phi(t)))
  with alpha_b = sum_k |A_kb|^2 + |C_kb|^2 (real) and gamma_b = 2 sum_k A_kb conj(C_kb).
  Same structure as y = |<a, x>|^2: the phase is lost, only intensities are measured.

  One cell gives only cos(phi + angle(gamma_b)), so the direction of motion is
  ambiguous. Neighbouring cells see the same phi(t) through different static
  references (different angle(gamma_b)); two or more cells with different angles
  give cos and sin, i.e. phi up to the usual trivial ambiguities:
    * global offset  phi -> phi + c   (gamma -> gamma * exp(-jc)): we only want changes
    * conjugation    phi -> -phi      (gamma -> conj(gamma)): sign of the motion

Method
  1. Spectral initialisation: the centred, scaled measurement matrix is (noise-free)
     rank 2. Its top-2 singular vectors trace an ellipse; mapping the ellipse to a
     circle gives phi(t) up to the ambiguities above.
  2. Gauss-Newton refinement of the full model (alpha_b, gamma_b, phi(t)) by
     nonlinear least squares, with a small smoothness term on phi (the target cannot
     jump between frames). Levenberg damping keeps the steps safe.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _unit_circle_from_trajectory(X: np.ndarray) -> np.ndarray:
    """Fit an ellipse a x^2 + b xy + c y^2 + d x + e y = 1 to 2D points, map it to the
    unit circle, return the unwrapped angle."""
    x, y = X[:, 0], X[:, 1]
    D = np.c_[x * x, x * y, y * y, x, y]
    a, b, c, d, e = np.linalg.lstsq(D, np.ones_like(x), rcond=None)[0]
    M = np.array([[a, b / 2], [b / 2, c]])
    w, V = np.linalg.eigh(M)
    if np.any(w <= 0):                      # not an ellipse: fall back to whitening
        Xc = X - X.mean(0)
        w, V = np.linalg.eigh(np.cov(Xc.T))
        if np.any(w <= 1e-12 * max(w.max(), 1e-300)):
            raise ValueError("degenerate trajectory: the data do not span two dimensions")
        Z = (Xc @ V) / np.sqrt(w)
    else:
        ctr = np.linalg.solve(2 * M, -np.array([d, e]))
        Z = ((X - ctr) @ V) * np.sqrt(w)
    return np.unwrap(np.arctan2(Z[:, 1], Z[:, 0]))


def spectral_init(Y: np.ndarray) -> np.ndarray:
    """Y: (T, B) linear power. Returns phi0 (T,).

    Needs at least two range cells whose signals vary independently: the centred data
    must have rank two, otherwise the phase cannot be recovered and ValueError is raised."""
    Y = np.asarray(Y, float)
    if Y.ndim != 2 or Y.shape[1] < 2 or Y.shape[0] < 5:
        raise ValueError("phase retrieval needs at least two range cells and five frames")
    Yc = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    U, S, _ = np.linalg.svd(Yc, full_matrices=False)
    if S[0] == 0 or S[1] <= 1e-6 * S[0]:
        raise ValueError("the selected cells do not carry two independent components "
                         "(rank below two); choose other cells")
    return _unit_circle_from_trajectory(U[:, :2] * S[:2])


def gauss_newton(Y: np.ndarray, phi0: np.ndarray, smooth: float = 1.0, iters: int = 30,
                 damping: float = 1e-3, verbose: bool = False):
    """Fit y_bt = alpha_b + gr_b cos(phi_t) - gi_b sin(phi_t) by Gauss-Newton.

    Y is scaled per cell (zero mean, unit std) so every cell weighs the same.
    smooth: weight of the residuals sqrt(smooth) * (phi_{t+1} - phi_t).
    Returns phi (T,), params dict, history of the cost."""
    T, B = Y.shape
    Ys = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    phi = phi0.astype(float).copy()
    # linear least squares for alpha, gamma given phi0
    alpha, gr, gi = np.zeros(B), np.zeros(B), np.zeros(B)

    def solve_linear(phi):
        A = np.c_[np.ones(T), np.cos(phi), -np.sin(phi)]
        coef = np.linalg.lstsq(A, Ys, rcond=None)[0]
        return coef[0], coef[1], coef[2]

    alpha, gr, gi = solve_linear(phi)
    sq = np.sqrt(smooth)
    nP = 3 * B + T
    hist = []

    def residual(alpha, gr, gi, phi):
        model = alpha[None, :] + np.cos(phi)[:, None] * gr[None, :] - np.sin(phi)[:, None] * gi[None, :]
        r_data = (model - Ys).ravel()                       # index t*B + b
        r_s = sq * np.diff(phi)
        return np.concatenate([r_data, r_s])

    r = residual(alpha, gr, gi, phi)
    cost = 0.5 * r @ r
    hist.append(cost)
    lam = damping
    tt, bb = np.meshgrid(np.arange(T), np.arange(B), indexing="ij")
    rows = (tt * B + bb).ravel()
    for it in range(iters):
        c, s = np.cos(phi), np.sin(phi)
        # Jacobian of data residuals
        cols, vals, rws = [], [], []
        rws.append(rows); cols.append(bb.ravel()); vals.append(np.ones(T * B))                    # d/d alpha_b
        rws.append(rows); cols.append(B + bb.ravel()); vals.append(np.repeat(c, B))                # d/d gr_b
        rws.append(rows); cols.append(2 * B + bb.ravel()); vals.append(-np.repeat(s, B))           # d/d gi_b
        dphi = (-s[:, None] * gr[None, :] - c[:, None] * gi[None, :]).ravel()                      # d/d phi_t
        rws.append(rows); cols.append(3 * B + tt.ravel()); vals.append(dphi)
        # smoothness rows
        k = np.arange(T - 1)
        rws.append(T * B + k); cols.append(3 * B + k + 1); vals.append(np.full(T - 1, sq))
        rws.append(T * B + k); cols.append(3 * B + k); vals.append(np.full(T - 1, -sq))
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rws), np.concatenate(cols))),
                          shape=(T * B + T - 1, nP))
        JTJ = (J.T @ J).tocsc()
        g = J.T @ r
        while True:
            H = JTJ + lam * sp.diags(JTJ.diagonal() + 1e-9)
            step = spla.spsolve(H, -g)
            a2, gr2, gi2 = alpha + step[:B], gr + step[B:2 * B], gi + step[2 * B:3 * B]
            phi2 = phi + step[3 * B:]
            r2 = residual(a2, gr2, gi2, phi2)
            cost2 = 0.5 * r2 @ r2
            if cost2 < cost:                    # accept, trust Gauss-Newton more
                alpha, gr, gi, phi, r, cost = a2, gr2, gi2, phi2, r2, cost2
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5                            # reject, damp more
            if lam > 1e6:
                break
        hist.append(cost)
        if verbose:
            print(f"iter {it}: cost {cost:.4f}  lambda {lam:.1e}")
        if len(hist) > 2 and abs(hist[-2] - hist[-1]) < 1e-7 * max(1.0, hist[-1]):
            break
    fit = 1 - np.sum(r[:T * B] ** 2) / np.sum(Ys ** 2)
    return phi, dict(alpha=alpha, gamma=gr + 1j * gi, fit_r2=float(fit)), hist


def displacement_mm(phi: np.ndarray, wavelength_m: float) -> np.ndarray:
    return phi * wavelength_m / (4 * np.pi) * 1e3


def recover(range_profile_db: np.ndarray, cells, wavelength_m: float, smooth: float = 1.0,
            iters: int = 30, verbose: bool = False):
    """Full pipeline on the board's range profile (frames, range_bins), dB.
    Returns displacement in mm (T,) relative to the start, the spectral-init
    displacement, and the fit info."""
    Y = 10 ** (np.asarray(range_profile_db, float)[:, list(cells)] / 10)
    phi0 = spectral_init(Y)
    phi, info, hist = gauss_newton(Y, phi0, smooth=smooth, iters=iters, verbose=verbose)
    d = displacement_mm(phi - phi[0], wavelength_m)
    d0 = displacement_mm(phi0 - phi0[0], wavelength_m)
    info["cost_history"] = hist
    return d, d0, info


# ---------------------------------------------------------------------------
# Log-domain antenna combination
# ---------------------------------------------------------------------------
# The demo firmware most likely does NOT add powers across antennas: its detection
# matrix sums log2 magnitudes over the virtual antennas, i.e. the range profile is
#     y_b(t) [dB] = (10/K) * sum_k log10( a_kb + Re(g_kb * exp(j*phi(t))) )
# (a geometric mean of the per-antenna powers). Then the "alpha + Re(gamma*u)" model
# above is wrong by construction (simulation: 0.1 mm error becomes 3.7 mm). This
# version fits the per-antenna model directly in dB, again by Gauss-Newton.

def gauss_newton_log(Ydb: np.ndarray, phi0: np.ndarray, K: int = 4, smooth: float = 1.0,
                     iters: int = 60, damping: float = 1e-2, seed: int = 0, verbose: bool = False):
    T, B = Ydb.shape
    rng = np.random.default_rng(seed)
    c10 = 10.0 / (K * np.log(10))
    # init: power-domain fit per cell, shared by all antennas, slightly perturbed
    Yp = 10 ** (Ydb / 10)
    A = np.c_[np.ones(T), np.cos(phi0), -np.sin(phi0)]
    coef = np.linalg.lstsq(A, Yp, rcond=None)[0]                 # (3, B)
    a = np.repeat(coef[0][:, None], K, 1) * (1 + 0.05 * rng.standard_normal((B, K)))
    gr = np.repeat(coef[1][:, None], K, 1) * (1 + 0.3 * rng.standard_normal((B, K)))
    gi = np.repeat(coef[2][:, None], K, 1) * (1 + 0.3 * rng.standard_normal((B, K)))
    mag = np.hypot(gr, gi)
    shrink = np.minimum(1.0, 0.8 * np.abs(a) / (mag + 1e-30))   # keep every antenna power positive
    a, gr, gi = np.abs(a), gr * shrink, gi * shrink
    phi = phi0.astype(float).copy()
    sq = np.sqrt(smooth)
    nAnt = B * K
    nP = 3 * nAnt + T + B          # + one dB offset per cell (firmware scaling is unknown)

    def powers(a, gr, gi, phi):
        c, s = np.cos(phi), np.sin(phi)
        P = a[None] + c[:, None, None] * gr[None] - s[:, None, None] * gi[None]   # (T,B,K)
        return np.maximum(P, 1e-12 * np.abs(a).max())

    def residual(a, gr, gi, phi, off):
        P = powers(a, gr, gi, phi)
        model = off[None, :] + c10 * np.log(P).sum(-1)                         # (T,B)
        return np.concatenate([(model - Ydb).ravel(), sq * np.diff(phi)]), P

    off = Ydb.mean(0) - (c10 * np.log(powers(a, gr, gi, phi)).sum(-1)).mean(0)
    r, P = residual(a, gr, gi, phi, off)
    cost = 0.5 * r @ r
    hist = [cost]
    lam = damping
    tt, bb, kk = np.meshgrid(np.arange(T), np.arange(B), np.arange(K), indexing="ij")
    rows = (tt * B + bb).ravel()
    pidx = (bb * K + kk).ravel()
    for it in range(iters):
        c, s = np.cos(phi), np.sin(phi)
        inv = (c10 / P).ravel()                                                # (T*B*K)
        cr = np.broadcast_to(c[:, None, None], P.shape).ravel()
        sr = np.broadcast_to(s[:, None, None], P.shape).ravel()
        dphi_full = (inv * (-sr * np.broadcast_to(gr[None], P.shape).ravel()
                            - cr * np.broadcast_to(gi[None], P.shape).ravel()))
        dphi = dphi_full.reshape(T, B, K).sum(-1).ravel()
        k1 = np.arange(T - 1)
        tb = (np.arange(T)[:, None] * B + np.arange(B)[None]).ravel()
        rr = np.concatenate([rows, rows, rows, tb, T * B + k1, T * B + k1, tb])
        cc = np.concatenate([pidx, nAnt + pidx, 2 * nAnt + pidx,
                             3 * nAnt + np.repeat(np.arange(T), B),
                             3 * nAnt + k1 + 1, 3 * nAnt + k1,
                             3 * nAnt + T + np.tile(np.arange(B), T)])
        vv = np.concatenate([inv, inv * cr, -inv * sr, dphi, np.full(T - 1, sq), np.full(T - 1, -sq),
                             np.ones(T * B)])
        J = sp.csr_matrix((vv, (rr, cc)), shape=(T * B + T - 1, nP))
        JTJ = (J.T @ J).tocsc()
        g = J.T @ r
        while True:
            H = JTJ + lam * sp.diags(JTJ.diagonal() + 1e-12)
            step = spla.spsolve(H, -g)
            a2 = a + step[:nAnt].reshape(B, K)
            gr2 = gr + step[nAnt:2 * nAnt].reshape(B, K)
            gi2 = gi + step[2 * nAnt:3 * nAnt].reshape(B, K)
            phi2 = phi + step[3 * nAnt:3 * nAnt + T]
            off2 = off + step[3 * nAnt + T:]
            r2, P2 = residual(a2, gr2, gi2, phi2, off2)
            cost2 = 0.5 * r2 @ r2
            if np.isfinite(cost2) and cost2 < cost:
                a, gr, gi, phi, off, r, P, cost = a2, gr2, gi2, phi2, off2, r2, P2, cost2
                lam = max(lam / 3, 1e-9)
                break
            lam *= 5
            if lam > 1e8:
                break
        hist.append(cost)
        if verbose:
            print(f"iter {it}: cost {cost:.4f}  lambda {lam:.1e}")
        if len(hist) > 3 and abs(hist[-2] - hist[-1]) < 1e-8 * max(1.0, hist[-1]):
            break
    rd = r[:T * B]
    Yc = Ydb - Ydb.mean(0)
    fit = 1 - np.sum(rd ** 2) / np.sum(Yc ** 2)
    return phi, dict(fit_r2=float(fit), cost_history=hist), hist


def recover_log(range_profile_db: np.ndarray, cells, wavelength_m: float, smooth: float = 1.0,
                iters: int = 60, seed: int = 0, verbose: bool = False):
    Ydb = np.asarray(range_profile_db, float)[:, list(cells)]
    phi0 = spectral_init(10 ** (Ydb / 10))
    phi, info, _ = gauss_newton_log(Ydb, phi0, smooth=smooth, iters=iters, seed=seed, verbose=verbose)
    return displacement_mm(phi - phi[0], wavelength_m), displacement_mm(phi0 - phi0[0], wavelength_m), info
