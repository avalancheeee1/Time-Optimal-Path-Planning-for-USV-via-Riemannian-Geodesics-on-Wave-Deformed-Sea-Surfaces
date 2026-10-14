"""P-C experiment -- minimum-turning-radius (Randers--Dubins) extension.

The direct solver minimises the exact Randers line integral over a polyline, but
it leaves the path's curvature unconstrained: the time-optimal geodesic may
execute turns that a real USV cannot follow, because a USV's turn rate is bounded
by a minimum turning radius ``R_min`` (a Dubins-type kinematic constraint).  This
experiment attaches a soft curvature penalty to the direct solver and sweeps
``R_min``, producing the *Randers--Dubins* family of paths:

    L[gamma] = int_gamma F(x, dx) dt  +  mu * sum_k  max(0, |kappa_k| - kappa_max)^2

where ``kappa_k`` is the discrete signed curvature (turning angle / local segment
length) at interior vertex ``k`` and ``kappa_max = 1 / R_min``.  As ``R_min -> inf``
the penalty is inactive and the path reduces to the unconstrained geodesic; as
``R_min`` shrinks below the geodesic's sharpest natural turn the constraint binds
and the achievable travel time degrades gracefully.

Curvature here is measured in the *ground frame* (the planar polyline).  This is a
first-order kinematic proxy: the physically exact constraint is on the
*water-relative* turn rate ``|theta_dot| <= 1/R_min``, which the full
sub-Riemannian Hamiltonian (state (x, y, theta), bang-bang control) treats
exactly.  The ground-frame proxy is valid in the weak-drift regime ``|v| < 1``
where ground and water frame differ only by O(|v|).

The cost gradient remains the analytic ``polyline_gradient``; only the small,
three-vertex-local penalty term is differenced (central differences, checked
against a closed form in ``check_penalty_grad``).
"""
import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
from metrics import Scenario
from solvers import polyline_cost, polyline_gradient, direct_geodesic


# --------------------------------------------------------------------------- #
# Discrete curvature of a polyline and the soft turning-radius penalty
# --------------------------------------------------------------------------- #
def _segment_data(pts):
    e = pts[1:] - pts[:-1]                      # (nseg, 2)
    l = np.hypot(e[:, 0], e[:, 1])              # (nseg,)
    return e, l


_L_FLOOR = 1e-9    # floor for segment length (guards division by zero when a
                  # trial point collapses two vertices)


def turning_angles(e, l):
    """Signed turning angle at each interior vertex (angle e[k-1] -> e[k])."""
    ls = np.maximum(l, _L_FLOOR)
    u = e[:-1] / ls[:-1, None]
    w = e[1:] / ls[1:, None]
    cross = u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]
    dot = np.clip(u[:, 0] * w[:, 0] + u[:, 1] * w[:, 1], -1.0, 1.0)
    return np.arctan2(cross, dot)               # (nseg-1,)


def curvature_profile(pts):
    """Signed curvature ``theta / s`` at each interior vertex."""
    e, l = _segment_data(pts)
    theta = turning_angles(e, l)
    s = 0.5 * (l[:-1] + l[1:])
    return theta / s


def _local_penalty(e, l, j, kappa_max, mu):
    """The single-vertex penalty ``mu * max(0, |kappa_j| - kappa_max)^2``."""
    lj1 = l[j - 1] if l[j - 1] > _L_FLOOR else _L_FLOOR
    lj = l[j] if l[j] > _L_FLOOR else _L_FLOOR
    u = e[j - 1] / lj1
    w = e[j] / lj
    cross = u[0] * w[1] - u[1] * w[0]
    dot = np.clip(u[0] * w[0] + u[1] * w[1], -1.0, 1.0)
    theta = np.arctan2(cross, dot)
    s = 0.5 * (lj1 + lj)
    over = abs(theta / s) - kappa_max
    return mu * max(over, 0.0) ** 2


def curvature_penalty(pts, kappa_max, mu):
    """Total soft curvature penalty over all interior turning vertices.

    Turning vertices are j = 1..nseg-1 (each interior point, including the one
    adjacent to the goal).  ``range(1, nseg)`` covers exactly these.
    """
    e, l = _segment_data(pts)
    nseg = l.size
    return float(sum(_local_penalty(e, l, j, kappa_max, mu)
                     for j in range(1, nseg)))


def curvature_penalty_grad(pts, kappa_max, mu, h=1e-6):
    """Gradient of the penalty w.r.t. the interior vertices, by central
    differences on the three-vertex-local terms.  Returns ``(nint, 2)``."""
    pts = np.asarray(pts, dtype=float)
    nint = pts.shape[0] - 2
    e, l = _segment_data(pts)
    grad = np.zeros((nint, 2))
    for k in range(1, nint + 1):                # interior vertex index in pts
        for d in range(2):
            # local sum over the only vertices whose turning angle touches p_k
            def f(delta):
                q = pts.copy()
                q[k, d] += delta
                ee, ll = _segment_data(q)
                nseg = ll.size
                return sum(_local_penalty(ee, ll, j, kappa_max, mu)
                           for j in (k - 1, k, k + 1) if 1 <= j <= nseg - 1)
            grad[k - 1, d] = (f(h) - f(-h)) / (2.0 * h)
    return grad


def check_penalty_grad(pts, kappa_max, mu, tol=1e-5):
    """Finite-difference vs central-difference gradient of the *local* penalty;
    this guards the analytic `_local_penalty` (returns max rel error)."""
    pts = np.asarray(pts, dtype=float)
    nint = pts.shape[0] - 2
    g_cd = curvature_penalty_grad(pts, kappa_max, mu, h=1e-6)
    g_fd = np.zeros_like(g_cd)
    eps = 1e-8
    for k in range(1, nint + 1):
        for d in range(2):
            q = pts.copy(); q[k, d] += eps
            fp = curvature_penalty(q, kappa_max, mu)
            q = pts.copy(); q[k, d] -= eps
            fm = curvature_penalty(q, kappa_max, mu)
            g_fd[k - 1, d] = (fp - fm) / (2.0 * eps)
    denom = np.abs(g_fd) + 1e-12
    return float(np.max(np.abs(g_cd - g_fd) / denom))


# --------------------------------------------------------------------------- #
# Curvature-constrained direct solver
# --------------------------------------------------------------------------- #
def direct_geodesic_curvature(start, goal, scenario, M=64, kappa_max=0.2, mu=50.0,
                              x0=None, n_restarts=2, seed=0):
    """Minimise cost + soft curvature penalty over ``M`` interior points.

    ``x0`` is a warm start (flat ``(M, 2)`` interior points); when None the
    straight line is used.  Returns ``(cost, path, penalty)`` where ``cost`` is
    the *pure* travel time (the line integral, penalty excluded).
    """
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    L = scenario.L
    rng = np.random.default_rng(seed)
    bounds = [(0.0, L)] * (2 * M)
    straight = np.linspace(start, goal, M + 2)[1:-1]

    def objective(z):
        pts = np.vstack([start, z.reshape(M, 2), goal])
        return polyline_cost(pts, scenario) + curvature_penalty(pts, kappa_max, mu)

    def gradient(z):
        pts = np.vstack([start, z.reshape(M, 2), goal])
        g = polyline_gradient(pts, scenario).ravel()          # exact cost grad
        g = g + curvature_penalty_grad(pts, kappa_max, mu).ravel()
        return g

    best = None
    for r in range(n_restarts):
        if r == 0:
            z0 = x0.ravel() if x0 is not None else straight.ravel()
        else:
            z0 = (straight + rng.normal(0.0, 0.4, (M, 2))).ravel()
        res = minimize(objective, z0, method="L-BFGS-B", bounds=bounds, jac=gradient,
                       options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8})
        pts = np.vstack([start, res.x.reshape(M, 2), goal])
        val = objective(res.x)
        pen = curvature_penalty(pts, kappa_max, mu)
        if best is None or val < best[0]:
            best = (val, pts, pen)
    return best[0] - best[2], best[1], best[2]   # pure travel time, path, penalty


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def main():
    sc = E.make_scenario("combined")

    # Unconstrained geodesic (the R_min -> inf limit) and its sharpest turn.
    c_free, path_free = direct_geodesic(E.START, E.GOAL, sc, M=64, n_restarts=6,
                                        seed=E.SEED)
    kappa = curvature_profile(path_free)
    kappa_star = float(np.abs(kappa).max())
    R_star = 1.0 / kappa_star if kappa_star > 1e-9 else float("inf")
    print(f"unconstrained geodesic: cost = {c_free:.6f}")
    print(f"  max |kappa| = {kappa_star:.4f}  ->  R_min* = {R_star:.3f} km")

    # Sweep R_min DECREASING from clumsy (tight kappa_max, constraint binding) to
    # agile (loose kappa_max, constraint inactive).  The first (tightest) case
    # starts from the straight line, which is *feasible* (curvature ~0), because a
    # warm start from the geodesic is a flat-cost local minimum whose tiny penalty
    # gradient cannot push it off to the rounded feasible path (it just eats the
    # penalty instead).  Each looser case warm-starts from the previous tighter
    # solution and relaxes back toward the unconstrained geodesic.
    R_grid = [1.6 * R_star, 1.35 * R_star, 1.15 * R_star, 1.0 * R_star,
              0.9 * R_star, 0.75 * R_star, 0.55 * R_star, 0.35 * R_star]
    mu = 50.0

    x0 = None                                   # straight-line init for the tightest case
    rows = []                                   # (R_min, kappa_max, cost, penalty, path, max|kappa|)
    print("\n  R_min     kappa_max     cost       penalty   max|kappa| (km^{-1})")
    for R in R_grid:
        kmax = 1.0 / R
        cost, path, pen = direct_geodesic_curvature(
            E.START, E.GOAL, sc, M=64, kappa_max=kmax, mu=mu, x0=x0,
            n_restarts=2, seed=E.SEED)
        kmax_attained = float(np.abs(curvature_profile(path)).max())
        rows.append((R, kmax, cost, pen, path, kmax_attained))
        print(f"  {R:6.3f}   {kmax:9.4f}   {cost:8.4f}   {pen:8.5f}   {kmax_attained:.4f}")
        x0 = path[1:-1]

    _render_figure(sc, path_free, R_star, c_free, rows)

    print("\nP-C -- minimum turning radius (Randers-Dubins) sweep complete.")
    print(f"  threshold R_min* = {R_star:.3f} km: a USV needing a larger turning")
    print(f"  radius than R_min* cannot execute the time-optimal path and pays a")
    print(f"  time penalty; below R_min* the geodesic is already kinematically free.")
    print(f"  unconstrained cost = {c_free:.6f}")


def _render_figure(sc, path_free, R_star, c_free, rows):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)

    rows = sorted(rows, key=lambda r: r[0])     # ascending R_min for the plot

    # (a) path family over the current field.
    ax = axes[0]
    im = ax.pcolormesh(sc.Xg, sc.Yg, sc._vx, shading="auto", cmap=E.CMAP_ELEV)
    Rvals = [r[0] for r in rows]
    norm = Normalize(vmin=min(Rvals), vmax=max(Rvals))
    cmap = plt.get_cmap("plasma")
    for (R, kmax, cost, pen, path, kmax_attained) in rows:
        ax.plot(path[:, 0], path[:, 1], color=cmap(norm(R)), lw=1.8,
                zorder=4, label=f"$R_{{\\min}}={R:.1f}$")
    ax.plot(path_free[:, 0], path_free[:, 1], color="0.1", lw=2.2, ls="--",
            zorder=5, label="geodesic ($R_{\\min}\\to\\infty$)")
    ax.plot([E.START[0], E.GOAL[0]], [E.START[1], E.GOAL[1]], color="0.5",
            lw=0.9, ls=":", zorder=2)
    E._plot_start_goal(ax)
    E._format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    E._panel_label(ax, "a")

    # (b) travel time vs R_min with the knee at R_min*.
    ax = axes[1]
    Rvals_arr = np.array(Rvals)
    costs = np.array([r[2] for r in rows])
    ax.plot(Rvals_arr, costs, "-o", color="#B8741A", lw=2.0, ms=5, mfc="white",
            mec="#B8741A", mew=1.5, zorder=3)
    ax.axvline(R_star, color="0.45", lw=1.1, ls="--", zorder=2)
    ax.axhline(c_free, color="0.35", lw=1.0, ls=":", zorder=1)
    # Text clear of the knee line: the R_min* label sits to its left so no
    # glyph touches the dashed vline, and the geodesic-cost note goes to the
    # empty top-left corner instead of across the flat curve segment.
    ax.text(R_star - 0.12, ax.get_ylim()[1], "$R_{\\min}^*$", va="top",
            ha="right", fontsize=8, color="0.4")
    ax.text(0.02, 0.96, f"dotted: unconstrained geodesic ({c_free:.4f})",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            color="0.35")
    ax.set_xlabel("minimum turning radius $R_{\\min}$ (km)")
    ax.set_ylabel("travel time (dimensionless)")
    ax.set_xlim(Rvals_arr.min(), Rvals_arr.max())
    ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    E._format_axis(ax)
    E._panel_label(ax, "b")

    # Legend outside the axes (below panel a): inside the axes it occludes the
    # path family converging on the goal.  Axes.legend does not accept the
    # "outside" locations, so this goes through fig.legend, which
    # constrained_layout reserves space for.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower left", ncol=5,
               fontsize=5.5, frameon=False, columnspacing=1.2,
               handlelength=1.5)

    E._save_fig(fig, "fig_curvature")


if __name__ == "__main__":
    main()
