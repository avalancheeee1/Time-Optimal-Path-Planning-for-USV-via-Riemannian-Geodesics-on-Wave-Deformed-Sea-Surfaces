"""Analytic gradients of the geodesic line integral.

The direct solver minimises

    T = sum_k sum_q w_q F(x_{k,q}, xi_k)

over the interior vertices ``p_k`` (``k = 1..M``), where ``xi_k = p_{k+1} - p_k``
and ``x_{k,q} = p_k + t_q xi_k`` are the Gauss--Legendre nodes on ``[0, 1]``.
The gradient of ``T`` w.r.t. an interior vertex ``p_k`` collects contributions
from its two incident segments:

    dT/dp_k = sum_q w_q [ dF/dx(x_{k-1,q}) * t_q + dF/dxi(x_{k-1,q}) ]       # seg k-1 (p_k is the end)
            + sum_q w_q [ dF/dx(x_{k,q}) * (1 - t_q) - dF/dxi(x_{k,q}) ]    # seg k   (p_k is the start)

Here ``dF/dx`` is the gradient of the local cost through the bilinearly
interpolated fields and ``dF/dxi`` the gradient through the displacement.  All
quantities are closed form (no finite differences), so supplying them to
L-BFGS-B removes the ``2M + 1`` cost evaluations per iteration that a
finite-difference Jacobian needs.

``verify_gradients.py`` checks every mode against central differences; it must
pass (agreement to ~1e-6) before this module is used in a production run.
"""

import numpy as np
from scipy.ndimage import map_coordinates


# --------------------------------------------------------------------------- #
# Bilinear-interpolation gradient (exact derivative of map_coordinates order=1)
# --------------------------------------------------------------------------- #
def _bilinear_grad(field, pts, dx):
    """Gradient ``(df/dx, df/dy)`` of bilinear interpolation of ``field`` (N,N).

    Matches the derivative of ``scipy.ndimage.map_coordinates(field, [i, j],
    order=1, mode="nearest")`` inside the domain (piecewise-constant per cell).
    """
    pts = np.asarray(pts, dtype=float)
    N = field.shape[0]
    g = pts / dx
    i = np.clip(g[:, 0], 0.0, N - 1.0)
    j = np.clip(g[:, 1], 0.0, N - 1.0)
    i0 = np.floor(i).astype(int)
    j0 = np.floor(j).astype(int)
    fx = i - i0
    fy = j - j0
    i1 = np.minimum(i0 + 1, N - 1)
    j1 = np.minimum(j0 + 1, N - 1)
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]
    df_di = (1.0 - fy) * (f10 - f00) + fy * (f11 - f01)
    df_dj = (1.0 - fx) * (f01 - f00) + fx * (f11 - f10)
    return df_di / dx, df_dj / dx


def _fields_grad(sc, pts):
    """Gradients of the interpolated fields ``(hx, hy, vx, vy)`` at ``pts``.

    Returns ``(dhx_dx, dhx_dy, dhy_dx, dhy_dy, dvx_dx, dvx_dy, dvy_dx, dvy_dy)``.
    """
    dhx_x, dhx_y = _bilinear_grad(sc._hx, pts, sc._dx)
    dhy_x, dhy_y = _bilinear_grad(sc._hy, pts, sc._dx)
    dvx_x, dvx_y = _bilinear_grad(sc._vx, pts, sc._dx)
    dvy_x, dvy_y = _bilinear_grad(sc._vy, pts, sc._dx)
    return dhx_x, dhx_y, dhy_x, dhy_y, dvx_x, dvx_y, dvy_x, dvy_y


def _wave_fields_grad(sc, pts):
    """Gradients of ``(psi, eps)`` at ``pts``: ``(dpsi_dx, dpsi_dy, deps_dx, deps_dy)``."""
    dpsi_x, dpsi_y = _bilinear_grad(sc._psi, pts, sc._dx)
    deps_x, deps_y = _bilinear_grad(sc._eps, pts, sc._dx)
    return dpsi_x, dpsi_y, deps_x, deps_y


# --------------------------------------------------------------------------- #
# Local cost + its partials, per mode
# --------------------------------------------------------------------------- #
def _surface_metric_grad(hx, hy):
    """Metric ``(gxx, gxy, gyy)`` and its partials w.r.t. ``(hx, hy)``."""
    gxx = 1.0 + hx * hx
    gxy = hx * hy
    gyy = 1.0 + hy * hy
    dgxx_dhx = 2.0 * hx
    dgxy_dhx = hy
    dgxx_dhy = 0.0
    dgxy_dhy = hx
    dgyy_dhy = 2.0 * hy
    return gxx, gxy, gyy, dgxx_dhx, dgxy_dhx, dgxx_dhy, dgxy_dhy, dgyy_dhy


def _randers_partials(xi_x, xi_y, vx, vy, gxx, gxy, gyy):
    """Randers (Zermelo) cost and its partials w.r.t. displacement and drift.

    ``F = (S - gxiv) / d`` with ``gxi = |xi|_g^2``, ``gv = |v|_g^2``,
    ``gxiv = <xi, v>_g``, ``d = 1 - gv``, ``S = sqrt(d * gxi + gxiv^2)``.

    Returns ``(F, dF_dxi_x, dF_dxi_y, dF_dvx, dF_dvy)``.  ``g`` is held fixed;
    the dependence of ``g`` on the surface gradient is handled by the caller via
    :func:`_surface_metric_grad`.
    """
    gxi = gxx * xi_x * xi_x + 2.0 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
    gv = gxx * vx * vx + 2.0 * gxy * vx * vy + gyy * vy * vy
    gxiv = gxx * xi_x * vx + gxy * (xi_x * vy + xi_y * vx) + gyy * xi_y * vy
    d = 1.0 - gv
    S = np.sqrt(d * gxi + gxiv * gxiv)
    F = (S - gxiv) / d

    # dF/dxi (manuscript eq. for the Randers metric).
    gxi_x = gxx * xi_x + gxy * xi_y      # (g . xi)_x
    gxi_y = gxy * xi_x + gyy * xi_y      # (g . xi)_y
    gv_x = gxx * vx + gxy * vy           # (g . v)_x
    gv_y = gxy * vx + gyy * vy           # (g . v)_y
    coef = (d * gxi + gxiv * gxiv) ** (-0.5)  # 1/S
    dF_dxi_x = ((d * gxi_x + gxiv * gv_x) * coef - gv_x) / d
    dF_dxi_y = ((d * gxi_y + gxiv * gv_y) * coef - gv_y) / d

    # dF/dv (xi, g fixed): dF/dv = [ (gxiv*(g.xi) - gxi*(g.v))/S - (g.xi) ]/d
    #                             + 2 F (g.v)/d
    dF_dv_x = ((gxiv * gxi_x - gxi * gv_x) * coef - gxi_x) / d + 2.0 * F * gv_x / d
    dF_dv_y = ((gxiv * gxi_y - gxi * gv_y) * coef - gxi_y) / d + 2.0 * F * gv_y / d
    return F, dF_dxi_x, dF_dxi_y, dF_dv_x, dF_dv_y


def cost_grad_batch(sc, pts, xi):
    """Local cost ``F`` (P,) and its position/direction gradients ``(dFdx, dFdxi)``.

    ``pts`` is (P, 2), ``xi`` is (P, 2).  ``dFdx`` and ``dFdxi`` are (P, 2).
    """
    pts = np.asarray(pts, dtype=float)
    xi = np.asarray(xi, dtype=float)
    P = xi.shape[0]
    xi_x, xi_y = xi[:, 0], xi[:, 1]
    mode = sc.mode

    if mode == "flat":
        n = np.hypot(xi_x, xi_y)
        n_safe = np.where(n > 1e-24, n, 1.0)
        F = n
        dFdxi = np.column_stack([xi_x / n_safe, xi_y / n_safe])
        return F, np.zeros_like(xi), dFdxi

    hx, hy, vx, vy = sc.fields(pts)

    if mode == "surface":
        gxx, gxy, gyy, *_ = _surface_metric_grad(hx, hy)
        gxi = gxx * xi_x * xi_x + 2.0 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
        F = np.sqrt(gxi)
        dFdxi_x = (gxx * xi_x + gxy * xi_y) / F
        dFdxi_y = (gxy * xi_x + gyy * xi_y) / F
        dFdxi = np.column_stack([dFdxi_x, dFdxi_y])
        # dF/dhx, dF/dhy
        dF_dhx = (hx * xi_x * xi_x + hy * xi_x * xi_y) / F
        dF_dhy = (hx * xi_x * xi_y + hy * xi_y * xi_y) / F
        dhx_x, dhx_y, dhy_x, dhy_y, *_ = _fields_grad(sc, pts)
        dFdx = np.column_stack([dF_dhx * dhx_x + dF_dhy * dhy_x,
                                dF_dhx * dhx_y + dF_dhy * dhy_y])
        return F, dFdx, dFdxi

    if mode == "current":
        gxx = np.ones_like(hx); gxy = np.zeros_like(hx); gyy = np.ones_like(hx)
        F, dFdxi_x, dFdxi_y, dF_dvx, dF_dvy = _randers_partials(
            xi_x, xi_y, vx, vy, gxx, gxy, gyy)
        dFdxi = np.column_stack([dFdxi_x, dFdxi_y])
        *_ , dvx_x, dvx_y, dvy_x, dvy_y = _fields_grad(sc, pts)
        dFdx = np.column_stack([dF_dvx * dvx_x + dF_dvy * dvy_x,
                                dF_dvx * dvx_y + dF_dvy * dvy_y])
        return F, dFdx, dFdxi

    if mode == "combined":
        gxx, gxy, gyy, dgxx_dhx, dgxy_dhx, dgxx_dhy, dgxy_dhy, dgyy_dhy = \
            _surface_metric_grad(hx, hy)
        F, dFdxi_x, dFdxi_y, dF_dvx, dF_dvy = _randers_partials(
            xi_x, xi_y, vx, vy, gxx, gxy, gyy)
        dFdxi = np.column_stack([dFdxi_x, dFdxi_y])

        # dF/dhx, dF/dhy via the chain rule through g_ij.  F = (S - gxiv)/d with
        # gxi, gv, gxiv, d, S all depending on g(hx, hy).
        gxi = gxx * xi_x * xi_x + 2.0 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
        gv = gxx * vx * vx + 2.0 * gxy * vx * vy + gyy * vy * vy
        gxiv = gxx * xi_x * vx + gxy * (xi_x * vy + xi_y * vx) + gyy * xi_y * vy
        d = 1.0 - gv
        S = np.sqrt(d * gxi + gxiv * gxiv)

        def dF_dh(pa):  # pa = partials (dgxx, dgxy, dgyy) of g w.r.t. h
            pxx, pxy, pyy = pa
            dgxi = pxx * xi_x * xi_x + 2.0 * pxy * xi_x * xi_y + pyy * xi_y * xi_y
            dgv = pxx * vx * vx + 2.0 * pxy * vx * vy + pyy * vy * vy
            dgxiv = pxx * xi_x * vx + pxy * (xi_x * vy + xi_y * vx) + pyy * xi_y * vy
            dd = -dgv
            dS = (d * dgxi + gxi * dd + 2.0 * gxiv * dgxiv) / (2.0 * S)
            return (dS - dgxiv) / d - (S - gxiv) * dd / (d * d)
        dF_dhx = dF_dh((dgxx_dhx, dgxy_dhx, 0.0))
        dF_dhy = dF_dh((dgxx_dhy, dgxy_dhy, dgyy_dhy))

        dhx_x, dhx_y, dhy_x, dhy_y, dvx_x, dvx_y, dvy_x, dvy_y = _fields_grad(sc, pts)
        dFdx = np.column_stack([
            dF_dhx * dhx_x + dF_dhy * dhy_x + dF_dvx * dvx_x + dF_dvy * dvy_x,
            dF_dhx * dhx_y + dF_dhy * dhy_y + dF_dvx * dvx_y + dF_dvy * dvy_y])
        return F, dFdx, dFdxi

    psi, eps = sc.wave_fields(pts)

    if mode == "wave":
        # F = |xi| / s(theta), s = 1 - eps (1 - cos(theta - psi)) / 2.
        n = np.hypot(xi_x, xi_y)
        theta = np.arctan2(xi_y, xi_x)
        s = 1.0 - eps * (1.0 - np.cos(theta - psi)) * 0.5
        F = n / s
        ds_dth = -0.5 * eps * np.sin(theta - psi)     # ds/dtheta
        ds_dpsi = 0.5 * eps * np.sin(theta - psi)     # ds/dpsi
        ds_deps = -0.5 * (1.0 - np.cos(theta - psi))  # ds/deps
        n_safe = np.where(n > 1e-24, n, 1.0)
        dth_dx = -xi_y / (n_safe * n_safe)            # dtheta/dxi_x
        dth_dy = xi_x / (n_safe * n_safe)             # dtheta/dxi_y
        # dF/dxi = xi/(s n) + n * (-1/s^2 ds/dtheta) * dtheta/dxi
        dF_dn = 1.0 / s
        dF_dth = -n * ds_dth / (s * s)
        dFdxi = np.column_stack([
            dF_dn * xi_x / n_safe + dF_dth * dth_dx,
            dF_dn * xi_y / n_safe + dF_dth * dth_dy])
        dF_dpsi = -n * ds_dpsi / (s * s)
        dF_deps = -n * ds_deps / (s * s)
        dpsi_x, dpsi_y, deps_x, deps_y = _wave_fields_grad(sc, pts)
        dFdx = np.column_stack([dF_dpsi * dpsi_x + dF_deps * deps_x,
                                dF_dpsi * dpsi_y + dF_deps * deps_y])
        return F, dFdx, dFdxi

    if mode == "wave_current":
        # F = T solves |xi/T - v| = s(theta(xi/T - v)).  Implicit function
        # theorem on g(T) = |w| - s(theta(w)) = 0 with w = xi/T - v.
        T = sc.cost_batch(pts, xi) if hasattr(sc, "cost_batch") else None
        # Recover T directly from the vectorised solver for a stable base point.
        from metrics import zermelo_finsler_time_vec
        T = zermelo_finsler_time_vec(xi, vx, vy, psi, eps)
        wx = xi_x / T - vx
        wy = xi_y / T - vy
        wmag = np.hypot(wx, wy)
        wmag_safe = np.where(wmag > 1e-24, wmag, 1.0)
        theta = np.arctan2(wy, wx)
        # dg/dw = w/|w| - (eps sin(theta-psi)/2) (w_y, -w_x)/|w|^2
        c = 0.5 * eps * np.sin(theta - psi)
        gw_x = wx / wmag_safe - c * wy / (wmag_safe * wmag_safe)
        gw_y = wy / wmag_safe + c * wx / (wmag_safe * wmag_safe)
        gw_dot_xi = gw_x * xi_x + gw_y * xi_y
        denom = np.where(np.abs(gw_dot_xi) > 1e-24, gw_dot_xi, 1.0)
        # dT/dxi = T * gw / (gw . xi);  dT/dv = -T^2 * gw / (gw . xi)
        # dT/dpsi = -T^2 (ds/dpsi) / (gw . xi);  dT/deps = -T^2 (ds/deps) / (gw . xi)
        ds_dpsi = 0.5 * eps * np.sin(theta - psi)
        ds_deps = -0.5 * (1.0 - np.cos(theta - psi))
        dFdxi = np.column_stack([T * gw_x / denom, T * gw_y / denom])
        dF_dvx = -T * T * gw_x / denom
        dF_dvy = -T * T * gw_y / denom
        dF_dpsi = -T * T * ds_dpsi / denom
        dF_deps = -T * T * ds_deps / denom
        dpsi_x, dpsi_y, deps_x, deps_y = _wave_fields_grad(sc, pts)
        *_ , dvx_x, dvx_y, dvy_x, dvy_y = _fields_grad(sc, pts)
        dFdx = np.column_stack([
            dF_dvx * dvx_x + dF_dvy * dvy_x + dF_dpsi * dpsi_x + dF_deps * deps_x,
            dF_dvx * dvx_y + dF_dvy * dvy_y + dF_dpsi * dpsi_y + dF_deps * deps_y])
        return T, dFdx, dFdxi

    raise ValueError(f"unsupported mode: {mode}")


# --------------------------------------------------------------------------- #
# Line-integral gradient w.r.t. interior vertices
# --------------------------------------------------------------------------- #
def polyline_gradient(points, scenario):
    """Gradient of ``polyline_cost`` w.r.t. the interior vertices ``(M, 2)``.

    ``points`` has shape ``(M+2, 2)`` with fixed endpoints.  Returns ``(M, 2)``.
    """
    pts = np.asarray(points, dtype=float)
    p0 = pts[:-1]
    d = pts[1:] - pts[:-1]              # (nseg, 2)
    nseg = d.shape[0]
    nint = nseg - 1                     # interior vertices p_1 .. p_{nseg-1}
    nq = 5                              # matches solvers._GL_T (5-point G--L)
    tq, wq = np.polynomial.legendre.leggauss(nq)
    tq = 0.5 * (tq + 1.0)               # nodes on [0, 1]
    wq = 0.5 * wq

    # Quadrature points: (nseg, nq, 2).
    seg_pts = p0[:, None, :] + tq[None, :, None] * d[:, None, :]
    seg_pts = seg_pts.reshape(nseg * nq, 2)
    xi = np.repeat(d, nq, axis=0)

    F, dFdx, dFdxi = cost_grad_batch(scenario, seg_pts, xi)
    dFdx = dFdx.reshape(nseg, nq, 2)
    dFdxi = dFdxi.reshape(nseg, nq, 2)
    t = tq[:, None]                 # (nq, 1)
    w = wq[:, None]                 # (nq, 1)

    # Interior vertex p_{j+1} (j = 0..nint-1) is the END of segment j and the
    # START of segment j+1.
    grad = np.zeros((nint, 2))
    for j in range(nint):
        grad[j] += (w * (dFdx[j] * t + dFdxi[j])).sum(axis=0)                 # end of seg j
        grad[j] += (w * (dFdx[j + 1] * (1.0 - t) - dFdxi[j + 1])).sum(axis=0)  # start of seg j+1
    return grad
