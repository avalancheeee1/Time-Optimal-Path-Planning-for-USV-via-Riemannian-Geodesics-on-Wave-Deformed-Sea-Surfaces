"""Fast-marching method (FMM) for the Randers (Zermelo) eikonal.

The time-optimal arrival time ``T`` solves the eikonal ``F*(x, grad T) = 1``,
where ``F*`` is the *dual* of the Randers norm ``F = alpha + beta``.  Under the
Zermelo correspondence ``F(x, xi) = 1  <=>  |xi - v(x)|_g = 1`` (the indicatrix of
``F`` is the unit ball of the surface metric ``g`` translated by the current
``v``), the dual has the closed form

    F*(x, p) = <p, v(x)> + sqrt(g^{ij}(x) p_i p_j),

with ``g`` the induced surface metric and ``v`` the current.  This is an upwind
finite-difference scheme, so unlike edge-priced grid Dijkstra it propagates the
front over a *continuum* of headings.

The local update combines two families of moves, both first-order consistent:

* **edge moves** -- the exact Randers cost ``F(x, x - y)`` of a straight hop to
  each already-finalised stencil neighbour ``y`` (midpoint metric, as in
  ``grid_dijkstra``); and
* **cardinal-pair moves** -- the Godunov solve ``F*(p) = 1`` with the upwind
  gradient built from one x- and one y-neighbour simultaneously, which lets the
  front advance along any heading inside a quadrant rather than only the
  stencil's discrete directions.

A pure 4-neighbour Godunov scheme overestimates diagonal propagation by O(h)
(the (1,1) node reads ``h + h/sqrt(2)`` instead of ``sqrt(2) h``); the diagonal
edge moves remove that bias.  The cardinal-pair solve still carries a signed
gradient per axis (the front arrives from whichever neighbour is smaller), and
that sign enters both the drift term ``<p, v>`` and the cross term ``g^{xy} p_x
p_y``.
"""

import heapq

import numpy as np

from metrics import surface_metric, randers_surface_cost
from selling import selling_decomposition_2d


# Stencil offsets (grid units).  Wider stencils shrink the residual directional
# (metrication) error, approaching the continuous optimum from above.
_STENCILS = {
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    8: [(-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)],
    24: [(di, dj) for di in (-2, -1, 0, 1, 2) for dj in (-2, -1, 0, 1, 2)
         if (di, dj) != (0, 0)],
}
_CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _solve_1d(G, v_signed):
    """Return ``D >= 0`` solving ``F*(s*D, 0) = 1`` on one axis.

    ``v_signed = s*v`` folds in the sign ``s`` of the upwind gradient component.
    Returns ``inf`` if the local update is infeasible.
    """
    denom = v_signed + np.sqrt(G)
    if denom <= 0.0:
        return np.inf
    return 1.0 / denom


def _solve_2d(Gxx, Gxy, Gyy, vx, vy, Tx, Ty, sx, sy, h):
    """Solve ``F*(p) = 1`` with ``p = (sx*(t-Tx)/h, sy*(t-Ty)/h)`` for ``t``.

    ``Tx``/``Ty`` are the upwind neighbour arrival times and ``sx``/``sy`` the
    signs of the two gradient components.  With the signed drift
    ``vt = (sx*vx, sy*vy)`` and signed cross term ``Gt = sx*sy*Gxy`` the equation
    reduces to a quadratic in ``t``.
    """
    vxt = sx * vx
    vyt = sy * vy
    Gxyt = sx * sy * Gxy
    Mxx = Gxx - vxt * vxt
    Mxy = Gxyt - vxt * vyt
    Myy = Gyy - vyt * vyt
    A = Mxx + 2.0 * Mxy + Myy
    B = 2.0 * (h * (vxt + vyt) - (Mxx + Mxy) * Tx - (Mxy + Myy) * Ty)
    C = (Mxx * Tx * Tx + 2.0 * Mxy * Tx * Ty + Myy * Ty * Ty
         - 2.0 * h * (vxt * Tx + vyt * Ty) - h * h)
    disc = B * B - 4.0 * A * C
    if A <= 0.0 or disc < 0.0:
        return np.inf
    t = (-B + np.sqrt(disc)) / (2.0 * A)
    if t >= Tx and t >= Ty:
        return t
    return np.inf


def _update(i, j, gxx, gxy, gyy, Gxx, Gxy, Gyy, vx, vy, T, state, h, stencil):
    """Tentative arrival time at node ``(i, j)`` from its finalised neighbours."""
    N_d = T.shape[0]
    best = np.inf

    # --- edge moves (exact Randers cost, midpoint metric) ---
    # The front arrives *from* (ni, nj) *to* (i, j), so the priced displacement
    # is (i - ni, j - nj) = (-di, -dj).  This sign matters: the Randers cost is
    # asymmetric, F(xi) != F(-xi) in general.
    for di, dj in stencil:
        ni, nj = i + di, j + dj
        if ni < 0 or ni >= N_d or nj < 0 or nj >= N_d or state[ni, nj] != 2:
            continue
        mgxx = 0.5 * (gxx[i, j] + gxx[ni, nj])
        mgxy = 0.5 * (gxy[i, j] + gxy[ni, nj])
        mgyy = 0.5 * (gyy[i, j] + gyy[ni, nj])
        mvx = 0.5 * (vx[i, j] + vx[ni, nj])
        mvy = 0.5 * (vy[i, j] + vy[ni, nj])
        cost = randers_surface_cost((-di * h, -dj * h), (mvx, mvy),
                                    mgxx, mgxy, mgyy)
        best = min(best, T[ni, nj] + cost)

    # --- cardinal-pair (Godunov) moves ---
    ux = vx[i, j]
    uy = vy[i, j]
    x_nei, y_nei = [], []
    if i > 0 and state[i - 1, j] == 2:
        x_nei.append((T[i - 1, j], +1))
    if i < N_d - 1 and state[i + 1, j] == 2:
        x_nei.append((T[i + 1, j], -1))
    if j > 0 and state[i, j - 1] == 2:
        y_nei.append((T[i, j - 1], +1))
    if j < N_d - 1 and state[i, j + 1] == 2:
        y_nei.append((T[i, j + 1], -1))
    for Tx, sx in x_nei:
        best = min(best, Tx + h * _solve_1d(Gxx[i, j], sx * ux))
    for Ty, sy in y_nei:
        best = min(best, Ty + h * _solve_1d(Gyy[i, j], sy * uy))
    for Tx, sx in x_nei:
        for Ty, sy in y_nei:
            best = min(best, _solve_2d(Gxx[i, j], Gxy[i, j], Gyy[i, j],
                                       ux, uy, Tx, Ty, sx, sy, h))
    return best


def fmm_randers(scenario, start, goal, N_d=193, stencil=8):
    """Upwind fast-marching solver for the Randers eikonal.

    Start and goal are snapped to the nearest grid node exactly as in
    ``grid_dijkstra`` so the two grid methods are compared like-for-like.

    Returns ``(cost, dist_field, xs, ys)``.
    """
    L = scenario.L
    h = L / (N_d - 1)
    xs = np.linspace(0.0, L, N_d)
    ys = np.linspace(0.0, L, N_d)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    hx, hy, vx, vy = scenario.fields(np.column_stack([Xg.ravel(), Yg.ravel()]))
    hx = hx.reshape(N_d, N_d)
    hy = hy.reshape(N_d, N_d)
    vx = vx.reshape(N_d, N_d)
    vy = vy.reshape(N_d, N_d)
    gxx, gxy, gyy, Gxx, Gxy, Gyy, _ = surface_metric(hx, hy)

    si = int(round(start[0] / h))
    sj = int(round(start[1] / h))
    gi = int(round(goal[0] / h))
    gj = int(round(goal[1] / h))
    si = min(max(si, 0), N_d - 1)
    sj = min(max(sj, 0), N_d - 1)
    gi = min(max(gi, 0), N_d - 1)
    gj = min(max(gj, 0), N_d - 1)

    T = np.full((N_d, N_d), np.inf)
    state = np.zeros((N_d, N_d), dtype=np.int8)    # 0 far, 1 band, 2 accepted
    T[si, sj] = 0.0
    state[si, sj] = 2
    heap = [(0.0, si, sj)]
    stencil = _STENCILS[stencil]

    while heap:
        d, i, j = heapq.heappop(heap)
        if d > T[i, j]:
            continue
        state[i, j] = 2
        for di, dj in _CARDINAL:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= N_d or nj < 0 or nj >= N_d or state[ni, nj] == 2:
                continue
            t = _update(ni, nj, gxx, gxy, gyy, Gxx, Gxy, Gyy, vx, vy,
                        T, state, h, stencil)
            if t < T[ni, nj]:
                T[ni, nj] = t
                state[ni, nj] = 1
                heapq.heappush(heap, (t, ni, nj))

    return float(T[gi, gj]), T, xs, ys


# --------------------------------------------------------------------------- #
# Lattice-basis-reduction (LBR) fast marching
# --------------------------------------------------------------------------- #
# The fixed cardinal/diagonal stencil of ``fmm_randers`` carries a directional
# (metrication) bias that only vanishes as the stencil widens.  The LBR variant
# replaces it with a *metric-adapted* stencil from Selling's lattice decomposition
# (``selling.py``), following Mirebeau (SIAM J. Numer. Anal. 52(4), 2014).
#
# The Randers dual eikonal ``F*(p) = <p, v> + sqrt(p^T G p) = 1`` is first cast
# into the equivalent shifted-Riemannian (Zermelo) form
#
#     (p - omega)^T H (p - omega) = R^2,
#     H = G - v v^T (SPD),   omega = -H^{-1} v,   R^2 = 1 + v^T H^{-1} v,
#
# and the SPD tensor ``H`` is decomposed by Selling as
#
#     H = sum_k nu_k * outer(off_k, off_k),    nu_k >= 0,  off_k integer.
#
# The upwind discretisation of the shifted eikonal is the 1-D decomposition scheme
#
#     sum_k nu_k * max(0, u - B_k)^2 = R^2,
#     B_k = min(U_minus + <omega, off_k>, U_plus - <omega, off_k>),
#
# where ``u = T / h`` is the scaled arrival time and ``U_minus``/``U_plus`` are the
# accepted neighbour values at ``(i,j) -/+ off_k``.  For an isotropic ``G`` the
# decomposition collapses to the two cardinals (the anti-diagonal carries zero
# weight), recovering the classic 4-neighbour Godunov update exactly; for an
# anisotropic ``G`` the offsets rotate to the metric's eigen-directions, removing
# the directional bias without widening the stencil.


def _lbr_solve(B, nu, R2):
    """Smallest ``u`` with ``sum_k nu_k * max(0, u - B_k)^2 = R2`` (monotone).

    ``B``/``nu`` are length-3 arrays; entries of ``B`` equal to ``+inf`` mark
    inactive offsets (no accepted neighbour in either direction).  Returns ``inf``
    when every offset is inactive.
    """
    finite = np.isfinite(B) & (nu > 0.0)
    if not finite.any():
        return np.inf
    Bf = B[finite]
    nuf = nu[finite]
    order = np.argsort(Bf)
    Bf = Bf[order]
    nuf = nuf[order]
    M = len(Bf)
    S0 = 0.0
    S1 = 0.0
    S2 = 0.0
    for m in range(M):
        S0 += nuf[m]
        S1 += nuf[m] * Bf[m]
        S2 += nuf[m] * Bf[m] * Bf[m]
        disc = S1 * S1 - S0 * (S2 - R2)
        if disc < 0.0:
            continue
        u = (S1 + np.sqrt(disc)) / S0
        if u >= Bf[m] and (m == M - 1 or u < Bf[m + 1]):
            return u
    return np.inf


def _lbr_update(i, j, nu, oi, oj, om_x, om_y, R2, T, h):
    """Scaled arrival ``u = T/h`` at ``(i, j)`` (LBR decomposition scheme).

    ``nu``/``oi``/``oj`` hold the per-node Selling coefficients and integer offsets
    of ``H = G - v v^T``; ``om_x``/``om_y`` the shift ``omega = -H^{-1} v``; ``R2``
    the right-hand side ``1 + v^T H^{-1} v``.  Unvisited neighbours (``T = inf``)
    fall out of the ``min`` automatically.
    """
    N_d = T.shape[0]
    B = np.full(3, np.inf)
    for k in range(3):
        di = oi[i, j, k]
        dj = oj[i, j, k]
        omf = om_x[i, j] * di + om_y[i, j] * dj
        um = np.inf
        up = np.inf
        ni, nj = i - di, j - dj
        if 0 <= ni < N_d and 0 <= nj < N_d:
            um = T[ni, nj] / h
        ni, nj = i + di, j + dj
        if 0 <= ni < N_d and 0 <= nj < N_d:
            up = T[ni, nj] / h
        B[k] = min(um + omf, up - omf)
    return _lbr_solve(B, nu[i, j], R2[i, j])


def fmm_randers_lbr(scenario, start, goal, N_d=193):
    """Lattice-basis-reduction solver for the Randers eikonal (fast sweeping).

    Solves ``F*(grad T) = 1`` with a per-node metric-adapted Selling stencil (the
    shifted-Riemannian reduction above) instead of a fixed stencil.  The shifted
    dual metric makes the update non-causal for a single-pass fast march, so the
    monotone scheme is iterated to a fixed point with four alternating Gauss-Seidel
    sweeps (fast sweeping), which converge to the viscosity solution.  Returns
    ``(cost, dist_field, xs, ys)``.
    """
    L = scenario.L
    h = L / (N_d - 1)
    xs = np.linspace(0.0, L, N_d)
    ys = np.linspace(0.0, L, N_d)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    hx, hy, vx, vy = scenario.fields(np.column_stack([Xg.ravel(), Yg.ravel()]))
    hx = hx.reshape(N_d, N_d)
    hy = hy.reshape(N_d, N_d)
    vx = vx.reshape(N_d, N_d)
    vy = vy.reshape(N_d, N_d)
    gxx, gxy, gyy, Gxx, Gxy, Gyy, _ = surface_metric(hx, hy)

    # Per-node shifted-Riemannian reduction and Selling decomposition of H.
    nu = np.zeros((N_d, N_d, 3))
    oi = np.zeros((N_d, N_d, 3), dtype=int)
    oj = np.zeros((N_d, N_d, 3), dtype=int)
    om_x = np.zeros((N_d, N_d))
    om_y = np.zeros((N_d, N_d))
    R2 = np.zeros((N_d, N_d))
    for i in range(N_d):
        for j in range(N_d):
            G = np.array([[Gxx[i, j], Gxy[i, j]], [Gxy[i, j], Gyy[i, j]]])
            v = np.array([vx[i, j], vy[i, j]])
            H = G - np.outer(v, v)
            det = H[0, 0] * H[1, 1] - H[0, 1] * H[0, 1]
            if H[0, 0] <= 0.0 or det <= 1e-14:
                # non-causal drift (|v|_g >= 1): leave node unreachable
                R2[i, j] = 0.0
                continue
            Hinv = np.linalg.inv(H)
            om = -Hinv @ v
            r2 = 1.0 + float(v @ Hinv @ v)
            coef, offs = selling_decomposition_2d(H)
            nu[i, j] = coef
            oi[i, j] = offs[0]
            oj[i, j] = offs[1]
            om_x[i, j] = om[0]
            om_y[i, j] = om[1]
            R2[i, j] = r2

    si = int(round(start[0] / h))
    sj = int(round(start[1] / h))
    gi = int(round(goal[0] / h))
    gj = int(round(goal[1] / h))
    si = min(max(si, 0), N_d - 1)
    sj = min(max(sj, 0), N_d - 1)
    gi = min(max(gi, 0), N_d - 1)
    gj = min(max(gj, 0), N_d - 1)

    T = np.full((N_d, N_d), np.inf)
    T[si, sj] = 0.0

    # Four alternating Gauss-Seidel sweeps; the shifted dual metric is non-causal,
    # so a single march does not suffice and the monotone scheme is iterated.
    orders = (
        (range(N_d), range(N_d)),
        (range(N_d - 1, -1, -1), range(N_d)),
        (range(N_d), range(N_d - 1, -1, -1)),
        (range(N_d - 1, -1, -1), range(N_d - 1, -1, -1)),
    )
    for _ in range(4 * N_d + 1):
        max_change = 0.0
        for I, J in orders:
            for i in I:
                row = T[i]
                for j in J:
                    if i == si and j == sj:
                        continue
                    t = _lbr_update(i, j, nu, oi, oj, om_x, om_y, R2, T, h)
                    t *= h
                    old = row[j]
                    if t < old:
                        max_change = max(max_change, old - t)
                        row[j] = t
        if max_change <= 1e-12:
            break

    return float(T[gi, gj]), T, xs, ys
