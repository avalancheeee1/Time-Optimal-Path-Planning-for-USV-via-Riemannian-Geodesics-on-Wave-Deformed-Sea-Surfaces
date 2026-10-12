"""P2 experiment -- geodesic ODE (two-point BVP) as an independent cross-check.

The direct solver minimises the exact Randers line integral over a polyline with
L-BFGS-B. That is a *global* optimisation; its weakness is that it could, in
principle, settle on a spurious local minimum of the discretised problem, or on a
discretisation-induced path.  This experiment supplies an independent,
methodologically distinct characterisation of the same time-optimal path.

For the flat-base Randers metric (the ``current`` mode), the geodesic is the
solution of the Zermelo navigation optimal-control problem.  Pontryagin's minimum
principle (maximum form) gives a first-order ODE in position ``x`` and costate
``lambda``:

    dx/dt   =  lambda / |lambda|  +  w(x)          (ground velocity)
    dlambda/dt = - (grad w(x))^T  lambda            (costate / adjoint)

where ``w(x)`` is the current drift and ``grad w`` its analytic Jacobian.  The
optimal heading is ``e = lambda / |lambda|`` (the unit water-relative velocity
points *along* the costate), and the free-final-time transversality condition
``H = 0`` fixes ``|lambda| + lambda . w = 1``.  A two-point boundary value problem
``x(0) = start``, ``x(T) = goal`` is solved by *shooting* on ``lambda(0)``: the
trajectory is integrated to its closest approach to the goal, and the miss
distance is driven to zero.  The travel time ``T`` is read off directly.

This ODE uses only the analytic drift field and its Jacobian -- no grid, no
interpolation, no line-integral quadrature -- so agreement with the direct solver
is a genuine independent verification of the headline optimal cost.
"""
import json

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
from metrics import Scenario
from solvers import direct_geodesic


S = E.START
G = E.GOAL

# --- analytic drift field + Jacobian (must match metrics.current_field) -------
U = 0.40
U0 = 0.15
SIGMA = 2.5
XC = 5.0
YC = 5.0


def _drift_jac(x):
    """Return ``(w, J)`` with ``w`` the drift vector and ``J = grad w`` its Jacobian."""
    x, y = x[0], x[1]
    dx = x - XC
    dy = y - YC
    r2 = dx * dx + dy * dy
    f = np.exp(-r2 / (2.0 * SIGMA * SIGMA))
    wx = -U * dy / SIGMA * f + U0
    wy = U * dx / SIGMA * f
    # Jacobian of (wx, wy) w.r.t. (x, y).
    dwxx = U * f * dx * dy / SIGMA ** 3
    dwxy = -U * f * (1.0 - dy * dy / SIGMA ** 2) / SIGMA
    dwyx = U * f * (1.0 - dx * dx / SIGMA ** 2) / SIGMA
    dwyy = -U * f * dx * dy / SIGMA ** 3
    return np.array([wx, wy]), np.array([[dwxx, dwxy], [dwyx, dwyy]])


def _ode(t, z):
    """RHS of the Zermelo geodesic ODE; ``z = [x, y, lambda_x, lambda_y]``."""
    x = z[0:2]
    lam = z[2:4]
    w, J = _drift_jac(x)
    n = np.hypot(lam[0], lam[1])
    xd = lam / n + w
    lamd = -(J.T @ lam)
    return np.concatenate([xd, lamd])


def _integrate(lam0, start, goal, Tmax=30.0):
    """Integrate the ODE from ``start`` with costate ``lam0`` and return the
    closest approach to ``goal``: ``(x_star, t_star, sol)``."""
    z0 = np.array([start[0], start[1], lam0[0], lam0[1]])
    sol = solve_ivp(_ode, [0.0, Tmax], z0, method="DOP853", rtol=1e-11, atol=1e-13,
                    dense_output=True)

    def d2(t):
        z = sol.sol(t)
        return (z[0] - goal[0]) ** 2 + (z[1] - goal[1]) ** 2

    # Coarse scan to bracket the minimum, then refine with Brent.
    ts = np.linspace(0.0, Tmax, 3001)
    ds = d2(ts)
    k = int(np.argmin(ds))
    lo = ts[max(0, k - 1)]
    hi = ts[min(len(ts) - 1, k + 1)]
    res = minimize_scalar(d2, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    tstar = float(res.x)
    xstar = sol.sol(tstar)[0:2]
    return xstar, tstar, sol


def _residual(lam0, start, goal):
    xstar, _tstar, _sol = _integrate(lam0, start, goal)
    return xstar - goal


def shoot_geodesic(start, goal, lam0_guess):
    """Solve the two-point BVP by shooting; returns ``(cost, lam0, res)``."""
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    res = least_squares(
        _residual, lam0_guess, args=(start, goal), method="trf",
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200,
    )
    lam0 = res.x
    xstar, tstar, sol = _integrate(lam0, start, goal)
    return float(tstar), lam0, res, sol


def _resample_by_arc(path, n=200):
    """Resample a (K, 2) path to ``n`` points uniformly by arc length."""
    path = np.asarray(path, float)
    d = np.diff(path, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    if s[-1] <= 0:
        return path
    s_new = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(s_new, s, path[:, 0]),
                            np.interp(s_new, s, path[:, 1])])


def main():
    sc_cur = Scenario(L=E.L, N=E.N_FIELD, mode="current")
    c_ref, p_ref = direct_geodesic(S, G, sc_cur, M=32, n_restarts=6, seed=E.SEED)

    # Initial guess for the costate: aim straight at the goal, de-biased by the
    # initial drift over a straight-line travel-time estimate.
    w0, _ = _drift_jac(S)
    straight = np.asarray(G, float) - np.asarray(S, float)
    lam0_guess = straight / np.hypot(*straight) - 0.5 * w0
    lam0_guess = lam0_guess / np.hypot(*lam0_guess)

    cost_ode, lam0, res, sol = shoot_geodesic(S, G, lam0_guess)

    # Extract the ODE geodesic as points from t=0 to t=T.
    ts = np.linspace(0.0, cost_ode, 400)
    zs = np.array([sol.sol(t) for t in ts])
    p_ode = zs[:, 0:2]

    # Path-level agreement: resample both to common arc length, max deviation.
    p_ref_rs = _resample_by_arc(p_ref)
    p_ode_rs = _resample_by_arc(p_ode)
    dev = np.hypot(*(p_ref_rs - p_ode_rs).T)
    max_dev = float(dev.max())
    mean_dev = float(dev.mean())

    rel_cost = (cost_ode - c_ref) / c_ref
    ok = (abs(rel_cost) < 5e-4) and (max_dev < 0.05)

    # Direct solver discretisation: as M grows its cost must converge to the
    # ODE (exact continuous) cost from above.
    M_grid = [8, 16, 32, 64]
    c_direct = np.array([
        direct_geodesic(S, G, sc_cur, M=M, n_restarts=6, seed=E.SEED)[0]
        for M in M_grid
    ])

    # Hamilton function H = |lambda| + lambda.w - 1 is *conserved* along the
    # geodesic (dH/dt = 0), independent of the arbitrary costate scale.
    H_res = np.array([
        np.hypot(zs[k, 2], zs[k, 3]) + zs[k, 2] * _drift_jac(zs[k, 0:2])[0][0]
        + zs[k, 3] * _drift_jac(zs[k, 0:2])[0][1]
        for k in range(len(ts))
    ])
    H_cons_err = float(np.abs(H_res - H_res[0]).max())

    results = {
        "direct_cost": c_ref,
        "ode_cost": cost_ode,
        "relative_cost_diff": rel_cost,
        "max_path_deviation": max_dev,
        "mean_path_deviation": mean_dev,
        "hamilton_conservation_error": H_cons_err,
        "convergence": {
            "M": M_grid, "direct_cost": c_direct.tolist(), "ode_cost": cost_ode,
        },
        "shooter": {
            "optimality": float(res.optimality),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "lam0": lam0.tolist(),
        },
    }
    with open("results_geodesic_ode.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _render_figure(p_ref, p_ode, sc_cur, M_grid, c_direct, cost_ode)

    print("P2 -- geodesic ODE two-point BVP (independent cross-check)")
    print(f"  direct solver cost (Randers 'current') = {c_ref:.6f}")
    print(f"  geodesic ODE   cost (Zermelo shooting) = {cost_ode:.6f}")
    print(f"  relative cost difference               = {rel_cost:+.3e}")
    print(f"  max  path deviation (km)               = {max_dev:.3e}")
    print(f"  mean path deviation (km)               = {mean_dev:.3e}")
    print(f"  Hamilton conservation error (max)      = {H_cons_err:.3e}")
    print(f"  direct cost vs M = {dict(zip(M_grid, np.round(c_direct, 6)))}")
    print(f"  shooter: optimality={res.optimality:.2e}, nfev={res.nfev}, "
          f"success={res.success}")
    print(f"  -> {'AGREES' if ok else 'DISAGREES'} with direct solver")


def _render_figure(p_ref, p_ode, sc_cur, M_grid, c_direct, cost_ode):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    ax = axes[0]
    im = ax.pcolormesh(sc_cur.Xg, sc_cur.Yg, sc_cur._vx, shading="auto",
                       cmap=E.CMAP_ELEV)
    ax.plot(p_ref[:, 0], p_ref[:, 1], color="#5E6AD2", lw=3.2, ls="--",
            label="direct solver", zorder=4)
    ax.plot(p_ode[:, 0], p_ode[:, 1], color="#1F9D55", lw=1.8,
            label="geodesic ODE (shooting)", zorder=5)
    E._plot_start_goal(ax)
    E._format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7)
    E._panel_label(ax, "a")

    # (b) direct-solver discretisation error closes toward the exact ODE cost.
    ax = axes[1]
    ax.set_xscale("log", base=2)
    ax.set_xticks(M_grid)
    ax.set_xticklabels([str(int(m)) for m in M_grid])
    ax.minorticks_off()
    ax.plot(M_grid, c_direct, "-o", color="#5E6AD2", lw=2.0, ms=5,
            mfc="white", mec="#5E6AD2", mew=1.5, label="direct solver", zorder=3)
    ax.axhline(cost_ode, color="#1F9D55", lw=1.6, ls="--",
               label="geodesic ODE (exact)", zorder=2)
    # Tight vertical range so the convergence, not the empty span, dominates.
    cmin = min(float(c_direct.min()), float(cost_ode))
    cmax = max(float(c_direct.max()), float(cost_ode))
    ax.set_ylim(cmin - 0.0008, cmax + 0.0008)
    ax.set_xlabel("interior points $M$")
    ax.set_ylabel("travel time")
    ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    E._format_axis(ax)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    E._panel_label(ax, "b")

    E._save_fig(fig, "fig_geodesic_ode")


if __name__ == "__main__":
    main()
