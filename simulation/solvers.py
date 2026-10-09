"""Numerical solvers for the geodesic / time-optimal planning problem.

Two solvers are provided:

* ``grid_dijkstra`` -- 8-connected grid search (Dijkstra / A* with admissible
  heuristic) over the exact local cost. Produces a full distance field plus a
  discrete geodesic. Used both as the "grid-search baseline" and to render the
  arrival-time / distance field.

* ``direct_geodesic`` -- continuous geodesic obtained by minimising the exact
  line integral ``integral F(gamma, gamma') dt`` of the (Randers) cost over a
  polyline. This is the proposed method; it is resolution-independent and
  handles the asymmetric Randers cost exactly.
"""

import heapq

import numpy as np
from scipy.optimize import minimize

from metrics import (
    riemannian_cost_vec,
    randers_flat_cost_vec,
    randers_surface_cost_vec,
    wave_speed_polar,
    zermelo_finsler_time,
)


# --------------------------------------------------------------------------- #
# Local cost from scalar fields (shared by Dijkstra edge evaluation)
# --------------------------------------------------------------------------- #
def _cost_from_fields(mode, hx, hy, vx, vy, xi_x, xi_y, psi, eps):
    if mode == "flat":
        return float(np.hypot(xi_x, xi_y))
    if mode == "surface":
        gxx = 1.0 + hx * hx
        gxy = hx * hy
        gyy = 1.0 + hy * hy
        return float(np.sqrt(gxx * xi_x * xi_x + 2 * gxy * xi_x * xi_y + gyy * xi_y * xi_y))
    if mode == "current":
        xi2 = xi_x * xi_x + xi_y * xi_y
        v2 = vx * vx + vy * vy
        vxi = vx * xi_x + vy * xi_y
        d = 1.0 - v2
        return float((np.sqrt(d * xi2 + vxi * vxi) - vxi) / d)
    if mode == "wave":
        theta = np.arctan2(xi_y, xi_x)
        s = wave_speed_polar(theta, psi, eps)
        return float(np.hypot(xi_x, xi_y) / s)
    if mode == "wave_current":
        return float(zermelo_finsler_time((xi_x, xi_y), (vx, vy), psi, eps))
    # combined
    gxx = 1.0 + hx * hx
    gxy = hx * hy
    gyy = 1.0 + hy * hy
    gxi = gxx * xi_x * xi_x + 2 * gxy * xi_x * xi_y + gyy * xi_y * xi_y
    gv = gxx * vx * vx + 2 * gxy * vx * vy + gyy * vy * vy
    gxiv = gxx * xi_x * vx + gxy * (xi_x * vy + xi_y * vx) + gyy * xi_y * vy
    d = 1.0 - gv
    return float((np.sqrt(d * gxi + gxiv * gxiv) - gxiv) / d)


# --------------------------------------------------------------------------- #
# Polyline cost (exact line integral via Gauss-Legendre quadrature)
# --------------------------------------------------------------------------- #
_GL_T, _GL_W = np.polynomial.legendre.leggauss(5)
_GL_T = 0.5 * (_GL_T + 1.0)   # nodes on [0, 1]
_GL_W = 0.5 * _GL_W           # weights on [0, 1]


def polyline_cost(points, scenario):
    """Total cost of a polyline ``points`` ((M+1, 2)) under ``scenario``."""
    pts = np.asarray(points, dtype=float)
    p0 = pts[:-1]
    d = pts[1:] - pts[:-1]              # (M, 2)
    M = d.shape[0]
    nq = _GL_T.shape[0]
    # All quadrature points at once: (M, nq, 2) -> (M*nq, 2).
    seg_pts = p0[:, None, :] + _GL_T[None, :, None] * d[:, None, :]
    seg_pts = seg_pts.reshape(M * nq, 2)
    xi = np.repeat(d, nq, axis=0)        # direction repeated per quadrature node
    c = scenario.cost_batch(seg_pts, xi).reshape(M, nq)
    return float(np.sum(c * _GL_W[None, :]))


def straight_line_cost(start, goal, scenario):
    return polyline_cost(np.array([start, goal]), scenario)


# --------------------------------------------------------------------------- #
# Direct geodesic (continuous optimization over the exact line integral)
# --------------------------------------------------------------------------- #
def direct_geodesic(start, goal, scenario, M=32, n_restarts=6, seed=0):
    """Minimise the Randers line integral over a polyline with ``M`` free
    interior points. Returns ``(cost, path)`` with ``path`` of shape ``(M+2, 2)``.
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    L = scenario.L
    rng = np.random.default_rng(seed)
    bounds = [(0.0, L)] * (2 * M)

    def objective(z):
        pts = np.vstack([start, z.reshape(M, 2), goal])
        return polyline_cost(pts, scenario)

    best_cost = np.inf
    best_path = None
    straight = np.linspace(start, goal, M + 2)[1:-1]

    for r in range(n_restarts):
        if r == 0:
            z0 = straight.ravel()
        else:
            z0 = (straight + rng.normal(0.0, 0.4, (M, 2))).ravel()
        res = minimize(
            objective, z0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7},
        )
        pts = np.vstack([start, res.x.reshape(M, 2), goal])
        c = polyline_cost(pts, scenario)
        if c < best_cost:
            best_cost = c
            best_path = pts
    return best_cost, best_path


# --------------------------------------------------------------------------- #
# Grid Dijkstra (8-connected) -- baseline + distance field
# --------------------------------------------------------------------------- #
_NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Wider stencils reduce the directional (metrication) error of the discrete grid
# scheme. 16-connected adds the eight knight moves; 24-connected is the full
# Chebyshev-radius-2 neighbourhood. Each edge (di, dj) is priced exactly through
# F(midpoint, (di*h, dj*h)), so a wider stencil approximates the continuous
# heading more faithfully without changing the edge-cost model.
_NEIGH16 = _NEIGH + [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                     (1, -2), (1, 2), (2, -1), (2, 1)]
_NEIGH24 = [(di, dj) for di in (-2, -1, 0, 1, 2) for dj in (-2, -1, 0, 1, 2)
            if (di, dj) != (0, 0)]
_NEIGHBORS = {8: _NEIGH, 16: _NEIGH16, 24: _NEIGH24}


def grid_dijkstra(scenario, start, goal, N_d=129, connectivity=8):
    """Run grid Dijkstra on an ``N_d x N_d`` grid.

    ``connectivity`` selects the stencil: 8 (default), 16, or 24 compass
    directions. Wider stencils reduce the directional (metrication) error of the
    discrete scheme for the anisotropic Randers metric.

    Returns ``(cost, path, dist_field, xs, ys)`` where ``dist_field`` is the
    arrival-time field from the start (shape ``(N_d, N_d)``), and ``path`` is the
    discrete geodesic (``(K, 2)``).

    Note: the Randers cost is asymmetric, so Dijkstra must propagate from the
    *start* and read ``dist[goal]`` (the start->goal cost); propagating from the
    goal would give the (different) goal->start cost.
    """
    L = scenario.L
    h = L / (N_d - 1)
    xs = np.linspace(0.0, L, N_d)
    ys = np.linspace(0.0, L, N_d)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    # Sample fields on the Dijkstra grid.
    hx, hy, vx, vy = scenario.fields(np.column_stack([Xg.ravel(), Yg.ravel()]))
    hx = hx.reshape(N_d, N_d)
    hy = hy.reshape(N_d, N_d)
    vx = vx.reshape(N_d, N_d)
    vy = vy.reshape(N_d, N_d)
    psi, eps = scenario.wave_fields(np.column_stack([Xg.ravel(), Yg.ravel()]))
    psi = psi.reshape(N_d, N_d)
    eps = eps.reshape(N_d, N_d)

    si = int(round(start[0] / h)); sj = int(round(start[1] / h))
    gi = int(round(goal[0] / h)); gj = int(round(goal[1] / h))
    si = min(max(si, 0), N_d - 1); sj = min(max(sj, 0), N_d - 1)
    gi = min(max(gi, 0), N_d - 1); gj = min(max(gj, 0), N_d - 1)

    def flat(i, j):
        return i + N_d * j

    src = flat(si, sj)          # propagate from START (asymmetric metric)
    tgt = flat(gi, gj)          # want distance at GOAL

    dist = np.full(N_d * N_d, np.inf)
    parent = np.full(N_d * N_d, -1, dtype=int)
    dist[src] = 0.0
    pq = [(0.0, src)]

    mode = scenario.mode
    neigh = _NEIGHBORS[connectivity]
    while pq:
        d, p = heapq.heappop(pq)
        if d > dist[p]:
            continue
        i = p % N_d
        j = p // N_d
        for di, dj in neigh:
            ni = i + di
            nj = j + dj
            if ni < 0 or ni >= N_d or nj < 0 or nj >= N_d:
                continue
            q = flat(ni, nj)
            dxi = di * h
            dyi = dj * h
            mhx = 0.5 * (hx[i, j] + hx[ni, nj])
            mhy = 0.5 * (hy[i, j] + hy[ni, nj])
            mvx = 0.5 * (vx[i, j] + vx[ni, nj])
            mvy = 0.5 * (vy[i, j] + vy[ni, nj])
            # Average the wave direction as a unit vector (angles wrap at +-pi);
            # averaging the raw angle would alias across the branch cut.
            cpsi = 0.5 * (np.cos(psi[i, j]) + np.cos(psi[ni, nj]))
            spsi = 0.5 * (np.sin(psi[i, j]) + np.sin(psi[ni, nj]))
            mpsi = np.arctan2(spsi, cpsi)
            meps = 0.5 * (eps[i, j] + eps[ni, nj])
            c = _cost_from_fields(mode, mhx, mhy, mvx, mvy, dxi, dyi, mpsi, meps)
            nd = d + c
            if nd < dist[q]:
                dist[q] = nd
                parent[q] = p
                heapq.heappush(pq, (nd, q))

    # Reconstruct path start -> goal by following parents back from the goal.
    path = []
    p = tgt
    guard = 0
    while p != -1 and guard < N_d * N_d:
        path.append((p % N_d * h, p // N_d * h))
        if p == src:
            break
        p = parent[p]
        guard += 1
    path = np.array(path[::-1])  # reverse -> start..goal

    cost = dist[tgt]
    dist_field = dist.reshape(N_d, N_d)
    return cost, path, dist_field, xs, ys
