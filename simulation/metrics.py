"""Metric and cost definitions for USV geodesic path planning.

A scenario provides a local cost ``F(x, xi)`` that is positively 1-homogeneous in
the displacement ``xi``. Three physical ingredients can be combined:

* **Surface metric** (waves): ``g_ij = delta_ij + (d_i h)(d_j h)`` for a graph
  surface ``z = h(x, y)`` embedded in R^3.
* **Current field** (drift): a spatially varying vector field ``v(x, y)``.
* **Zermelo navigation -> Randers metric**: time-optimal navigation on a
  Riemannian base metric ``g`` with drift ``v`` is governed by the Randers metric
  ``F = alpha + beta`` below, valid under the weak-drift condition ``|v|_g < 1``.

All quantities are dimensionless; the paper interprets the domain as a
``L x L`` patch (km).
"""

import numpy as np
from scipy.ndimage import map_coordinates


# --------------------------------------------------------------------------- #
# Surface geometry
# --------------------------------------------------------------------------- #
def surface_height(X, Y, A=(0.5, 0.3), lam=(10.0, 5.0), phi=0.5):
    """Elevation field h(x, y) (sum of two swell components)."""
    A1, A2 = A
    l1, l2 = lam
    return (
        A1 * np.sin(2 * np.pi * X / l1) * np.cos(2 * np.pi * Y / l1)
        + A2 * np.sin(2 * np.pi * X / l2 + phi) * np.sin(2 * np.pi * Y / l2)
    )


def surface_grad(X, Y, A=(0.5, 0.3), lam=(10.0, 5.0), phi=0.5):
    """Gradient (hx, hy) of the elevation field."""
    A1, A2 = A
    l1, l2 = lam
    k1 = 2 * np.pi / l1
    k2 = 2 * np.pi / l2
    hx = (
        A1 * k1 * np.cos(k1 * X) * np.cos(k1 * Y)
        + A2 * k2 * np.cos(k2 * X + phi) * np.sin(k2 * Y)
    )
    hy = (
        -A1 * k1 * np.sin(k1 * X) * np.sin(k1 * Y)
        + A2 * k2 * np.sin(k2 * X + phi) * np.cos(k2 * Y)
    )
    return hx, hy


# --------------------------------------------------------------------------- #
# Current field
# --------------------------------------------------------------------------- #
def current_field(X, Y, U=0.40, U0=0.15, xc=5.0, yc=5.0, sigma=2.5):
    """A smooth Rankine-like eddy plus a uniform background current.

    The eddy rotates counter-clockwise; ``U0`` is a rightward background drift.
    Parameters are chosen so that ``max |v| < 1`` everywhere (weak drift).
    """
    r2 = (X - xc) ** 2 + (Y - yc) ** 2
    f = np.exp(-r2 / (2 * sigma * sigma))
    vx = -U * (Y - yc) / sigma * f + U0
    vy = U * (X - xc) / sigma * f
    return vx, vy


# --------------------------------------------------------------------------- #
# Irregular spectral sea (Pierson--Moskowitz-type superposition)
# --------------------------------------------------------------------------- #
def surface_field_spectral(X, Y, seed=0, n_wave=32, n_dir=8, Hs=0.5, kp=0.9):
    """Irregular sea surface from a directional Pierson--Moskowitz-type spectrum.

    Returns ``(h, hx, hy)`` -- the elevation and its analytic gradient. The
    surface is a superposition of linear wave components whose amplitudes follow
    a PM-type wavenumber density ``S(k) ~ k^-3 exp(-1.25 (kp/k)^2)``, sampled at
    ``n_wave`` wavenumbers and ``n_dir`` directions with random phases, then
    renormalised to a target significant amplitude ``Hs`` (Hs = 4 * rms). This is
    a *realisation* of a realistic spectrum, not a specific observed sea state.
    """
    rng = np.random.default_rng(seed)
    k = np.linspace(0.35, 3.2, n_wave)                 # wavenumbers 2*pi/lambda
    S = k ** -3.0 * np.exp(-1.25 * (kp / k) ** 2)      # PM-type density
    theta = np.linspace(0.0, 2.0 * np.pi, n_dir, endpoint=False)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_wave, n_dir))
    dk = k[1] - k[0]
    dtheta = theta[1] - theta[0]

    h = np.zeros_like(X)
    hx = np.zeros_like(X)
    hy = np.zeros_like(X)
    var = 0.0
    for i in range(n_wave):
        ki = k[i]
        for j in range(n_dir):
            cx = np.cos(theta[j])
            sy = np.sin(theta[j])
            amp = np.sqrt(2.0 * S[i] * dk * dtheta)
            arg = ki * (X * cx + Y * sy) + phase[i, j]
            h += amp * np.cos(arg)
            hx += -amp * ki * cx * np.sin(arg)
            hy += -amp * ki * sy * np.sin(arg)
            var += 0.5 * amp * amp

    # Renormalise to the target significant amplitude (Hs = 4 * rms).
    rms = np.sqrt(var) if var > 0 else 0.0
    scale = (Hs / (4.0 * rms)) if rms > 1e-12 else 1.0
    return h * scale, hx * scale, hy * scale


# --------------------------------------------------------------------------- #
# Metric tensors
# --------------------------------------------------------------------------- #
def surface_metric(hx, hy):
    """Induced metric and its inverse for the graph surface z = h(x, y).

    Returns ``(gxx, gxy, gyy, Gxx, Gxy, Gyy, sqrt_det)``.
    """
    gxx = 1.0 + hx * hx
    gxy = hx * hy
    gyy = 1.0 + hy * hy
    det = 1.0 + hx * hx + hy * hy  # = det(g)
    Gxx = 1.0 - hx * hx / det      # g^{xx}
    Gxy = -hx * hy / det           # g^{xy}
    Gyy = 1.0 - hy * hy / det      # g^{yy}
    return gxx, gxy, gyy, Gxx, Gxy, Gyy, np.sqrt(det)


# --------------------------------------------------------------------------- #
# Randers (Zermelo) cost
# --------------------------------------------------------------------------- #
def randers_flat_cost(xi, v):
    """Randers cost on the flat base metric with current ``v``.

    ``xi`` is the displacement vector; ``v`` is the local current (|v| < 1).
    """
    xi_x, xi_y = xi
    vx, vy = v
    xi2 = xi_x * xi_x + xi_y * xi_y
    v2 = vx * vx + vy * vy
    vxi = vx * xi_x + vy * xi_y
    d = 1.0 - v2
    return (np.sqrt(d * xi2 + vxi * vxi) - vxi) / d


def randers_surface_cost(xi, v, gxx, gxy, gyy):
    """Randers cost on a Riemannian base metric ``g`` with current ``v``.

    ``xi`` and ``v`` are expressed in the ``(x, y)`` coordinate basis of the
    surface parameterisation; inner products use the surface metric ``g``.
    """
    xi_x, xi_y = xi
    vx, vy = v
    gxi = gxx * xi_x * xi_x + 2 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
    gv = gxx * vx * vx + 2 * gxy * vx * vy + gyy * vy * vy
    gxiv = gxx * xi_x * vx + gxy * (xi_x * vy + xi_y * vx) + gyy * xi_y * vy
    d = 1.0 - gv
    return (np.sqrt(d * gxi + gxiv * gxiv) - gxiv) / d


def riemannian_cost(xi, gxx, gxy, gyy):
    """Pure Riemannian arc-length cost ``sqrt(g_ij xi^i xi^j)`` (no current)."""
    xi_x, xi_y = xi
    return np.sqrt(
        gxx * xi_x * xi_x + 2 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
    )


# --------------------------------------------------------------------------- #
# Scenario wrapper: provides local cost F(x, xi) with grid interpolation
# --------------------------------------------------------------------------- #
class Scenario:
    """Encapsulates one physical scenario and exposes ``cost(x, xi)``.

    Parameters
    ----------
    L : float
        Domain is ``[0, L] x [0, L]``.
    N : int
        Interpolation grid resolution (N x N).
    mode : str
        One of ``"flat"`` (Euclidean, no wave/current), ``"surface"``,
        ``"current"``, ``"combined"``.
    """

    def __init__(self, L=10.0, N=257, mode="combined",
                 A=(0.5, 0.3), lam=(10.0, 5.0), phi=0.5,
                 U=0.40, U0=0.15, sigma=2.5, xc=5.0, yc=5.0):
        self.L = L
        self.N = N
        self.mode = mode

        xs = np.linspace(0.0, L, N)
        ys = np.linspace(0.0, L, N)
        self.Xg, self.Yg = np.meshgrid(xs, ys, indexing="ij")  # [i, j]

        # Precompute fields on the grid.
        self._hx = np.zeros_like(self.Xg)
        self._hy = np.zeros_like(self.Xg)
        self._vx = np.zeros_like(self.Xg)
        self._vy = np.zeros_like(self.Xg)
        if mode in ("surface", "combined"):
            self._hx, self._hy = surface_grad(self.Xg, self.Yg, A=A, lam=lam, phi=phi)
            self.h = surface_height(self.Xg, self.Yg, A=A, lam=lam, phi=phi)
        else:
            self.h = np.zeros_like(self.Xg)
        if mode in ("current", "combined"):
            self._vx, self._vy = current_field(
                self.Xg, self.Yg, U=U, U0=U0, sigma=sigma, xc=xc, yc=yc)

        self._dx = L / (N - 1)

    def fields(self, pts):
        """Return (hx, hy, vx, vy) at arbitrary points ``pts`` (M, 2) via linear
        (C-speed) grid interpolation."""
        pts = np.asarray(pts, dtype=float)
        g = pts / self._dx           # physical -> grid index (i = x, j = y)
        i = g[:, 0]
        j = g[:, 1]
        hx = map_coordinates(self._hx, [i, j], order=1, mode="nearest")
        hy = map_coordinates(self._hy, [i, j], order=1, mode="nearest")
        vx = map_coordinates(self._vx, [i, j], order=1, mode="nearest")
        vy = map_coordinates(self._vy, [i, j], order=1, mode="nearest")
        return hx, hy, vx, vy

    def cost(self, x, xi):
        """Local cost F(x, xi) at point x (2,) with displacement xi (2,)."""
        p = np.asarray([x], dtype=float)
        hx, hy, vx, vy = self.fields(p)
        hx, hy, vx, vy = float(hx[0]), float(hy[0]), float(vx[0]), float(vy[0])

        if self.mode == "flat":
            return float(np.hypot(xi[0], xi[1]))
        if self.mode == "surface":
            gxx, gxy, gyy, *_ = surface_metric(hx, hy)
            return float(riemannian_cost(xi, gxx, gxy, gyy))
        if self.mode == "current":
            return float(randers_flat_cost(xi, (vx, vy)))
        # combined
        gxx, gxy, gyy, *_ = surface_metric(hx, hy)
        return float(randers_surface_cost(xi, (vx, vy), gxx, gxy, gyy))

    def cost_batch(self, pts, xi):
        """Vectorized cost for arrays ``pts`` (M,2), ``xi`` (M,2)."""
        hx, hy, vx, vy = self.fields(pts)
        if self.mode == "flat":
            return np.hypot(xi[:, 0], xi[:, 1])
        if self.mode == "surface":
            gxx, gxy, gyy, *_ = surface_metric(hx, hy)
            return riemannian_cost_vec(xi, gxx, gxy, gyy)
        if self.mode == "current":
            return randers_flat_cost_vec(xi, vx, vy)
        gxx, gxy, gyy, *_ = surface_metric(hx, hy)
        return randers_surface_cost_vec(xi, vx, vy, gxx, gxy, gyy)


def riemannian_cost_vec(xi, gxx, gxy, gyy):
    xi_x, xi_y = xi[:, 0], xi[:, 1]
    return np.sqrt(gxx * xi_x * xi_x + 2 * gxy * xi_x * xi_y + gyy * xi_y * xi_y)


def randers_flat_cost_vec(xi, vx, vy):
    xi_x, xi_y = xi[:, 0], xi[:, 1]
    xi2 = xi_x * xi_x + xi_y * xi_y
    v2 = vx * vx + vy * vy
    vxi = vx * xi_x + vy * xi_y
    d = 1.0 - v2
    return (np.sqrt(d * xi2 + vxi * vxi) - vxi) / d


def randers_surface_cost_vec(xi, vx, vy, gxx, gxy, gyy):
    xi_x, xi_y = xi[:, 0], xi[:, 1]
    gxi = gxx * xi_x * xi_x + 2 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
    gv = gxx * vx * vx + 2 * gxy * vx * vy + gyy * vy * vy
    gxiv = gxx * xi_x * vx + gxy * (xi_x * vy + xi_y * vx) + gyy * xi_y * vy
    d = 1.0 - gv
    return (np.sqrt(d * gxi + gxiv * gxiv) - gxiv) / d
