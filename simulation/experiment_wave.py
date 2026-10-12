"""P1 experiment -- heading-dependent wave added resistance as a Finsler metric.

The paper's ``surface`` metric treats the wave field as *pure geometry*: the
induced Riemannian arc length ``g_ij = delta_ij + (d_i h)(d_j h)`` is symmetric,
so travelling up- and down-swell lengthens the path equally.  Real waves do not
just lengthen the path -- they slow a vessel in head seas and speed it in
following seas.  This experiment replaces that symmetric arc length with a
*heading-dependent* Finsler metric

    F(x, xi) = |xi| / s(x, theta(xi)),
    s(x, theta) = 1 - eps(x) * (1 - cos(theta - psi(x))) / 2,

where ``psi(x)`` is the local wave propagation direction and ``eps(x) in [0, EPS0]``
the local speed-reduction amplitude (proportional to wave steepness).  The full
Zermelo combination with the current drift (the ``wave_current`` mode) is the
general Finsler metric of which the paper's flat Randers ``current`` metric is
the isotropic-speed limit (``eps0 -> 0``).

Two apples-to-apples decompositions isolate the heading-dependence:

* ``surface`` (geometry only) vs ``wave`` (directional resistance only), no
  current -- the core novelty.
* ``current`` (isotropic Randers) vs ``wave_current`` (heading-dependent Zermelo
  Finsler), both with drift -- the extension to the drift case, with the
  ``eps0 -> 0`` reduction checked explicitly.

Run from the ``simulation/`` directory:

    python experiment_wave.py

Outputs
-------
- ``figures/fig_wave_metrics.{svg,pdf,png}``
- ``results_wave.json`` (machine-readable numbers for the manuscript).
"""
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
from metrics import Scenario, EPS0
from solvers import direct_geodesic, straight_line_cost, polyline_cost


S = E.START
G = E.GOAL

C_ARC = "#5E6AD2"    # arc-length (surface) geodesic -- indigo
C_WAV = "#B8741A"    # heading-dependent (wave) geodesic -- dark amber


def _ground_length(path):
    """Euclidean (planar ground-projected) arc length of a path."""
    d = np.diff(path, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _travel_time(path, sc):
    """Metric cost of a path under scenario ``sc`` (the actual travel time)."""
    return float(polyline_cost(path, sc))


def _added_resistance(path, sc_wave):
    """Extra travel time due to directional resistance: time - ground length.

    On calm water (s == 1) the two coincide, so this isolates the wave slowdown."""
    return _travel_time(path, sc_wave) - _ground_length(path)


def main():
    sc_surf = Scenario(L=E.L, N=E.N_FIELD, mode="surface")
    sc_wave = Scenario(L=E.L, N=E.N_FIELD, mode="wave")
    sc_cur = Scenario(L=E.L, N=E.N_FIELD, mode="current")
    sc_wc = Scenario(L=E.L, N=E.N_FIELD, mode="wave_current")

    c_surf, p_surf = direct_geodesic(S, G, sc_surf, M=32, n_restarts=6, seed=E.SEED)
    c_wave, p_wave = direct_geodesic(S, G, sc_wave, M=32, n_restarts=6, seed=E.SEED)
    c_cur, p_cur = direct_geodesic(S, G, sc_cur, M=32, n_restarts=6, seed=E.SEED)
    c_wc, p_wc = direct_geodesic(S, G, sc_wc, M=32, n_restarts=6, seed=E.SEED)

    # --- Wave-only decomposition: ground distance + added resistance ----------
    surf_time = _travel_time(p_surf, sc_wave)
    surf_ground = _ground_length(p_surf)
    surf_res = surf_time - surf_ground
    wave_time = c_wave
    wave_ground = _ground_length(p_wave)
    wave_res = wave_time - wave_ground

    # --- With drift: isotropic Randers vs heading-dependent Zermelo Finsler ---
    # eps0 -> 0 reduction check: wave_current must collapse onto `current`.
    sc_wc0 = Scenario(L=E.L, N=E.N_FIELD, mode="wave_current", eps0=0.0)
    c_wc0, _ = direct_geodesic(S, G, sc_wc0, M=32, n_restarts=6, seed=E.SEED)

    cur_under_wc = _travel_time(p_cur, sc_wc)      # Randers path, actual time
    wc_under_cur = _travel_time(p_wc, sc_cur)      # Finsler path under Randers

    time_penalty = (surf_time - wave_time) / surf_time
    res_reduction = (surf_res - wave_res) / surf_res
    len_penalty = (wave_ground - surf_ground) / surf_ground
    drift_penalty = (cur_under_wc - c_wc) / cur_under_wc

    results = {
        "eps0": EPS0,
        "wave_only": {
            "surface_path": {
                "arc_length": c_surf, "ground_length": surf_ground,
                "added_resistance": surf_res, "travel_time_in_waves": surf_time,
                "path": p_surf.tolist(),
            },
            "wave_path": {
                "travel_time": wave_time, "ground_length": wave_ground,
                "added_resistance": wave_res, "path": p_wave.tolist(),
            },
            "wrong_metric_time_loss": time_penalty,
            "added_resistance_reduction": res_reduction,
            "ground_length_premium": len_penalty,
        },
        "with_drift": {
            "randers_current_cost": c_cur,
            "wave_current_eps0_zero_cost": c_wc0,
            "wave_current_cost": c_wc,
            "randers_path_under_wave_current": cur_under_wc,
            "wave_current_path_under_randers": wc_under_cur,
            "wrong_metric_time_loss": drift_penalty,
        },
    }

    # Sea-severity sweep: how does the wrong-metric penalty grow with eps0?
    eps_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    penalties = np.zeros_like(eps_grid)
    res_reductions = np.zeros_like(eps_grid)
    for k, e0 in enumerate(eps_grid):
        scw = Scenario(L=E.L, N=E.N_FIELD, mode="wave", eps0=float(e0))
        cw_k, pw_k = direct_geodesic(S, G, scw, M=32, n_restarts=6, seed=E.SEED)
        surf_time_k = _travel_time(p_surf, scw)
        surf_res_k = surf_time_k - surf_ground
        wave_res_k = cw_k - _ground_length(pw_k)
        penalties[k] = (surf_time_k - cw_k) / surf_time_k
        res_reductions[k] = (surf_res_k - wave_res_k) / surf_res_k

    results["severity_sweep"] = {
        "eps0": eps_grid.tolist(),
        "wrong_metric_time_loss": penalties.tolist(),
        "added_resistance_reduction": res_reductions.tolist(),
    }

    with open("results_wave.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _render_figure(sc_wave, p_surf, p_wave, eps_grid, penalties)

    print("P1 -- heading-dependent Finsler metric (wave added resistance)")
    print(f"  peak head-sea slowdown eps0        = {EPS0:.2f}")
    print(f"  [wave only] surface geodesic arc length = {c_surf:.4f}")
    print(f"    ground length                    = {surf_ground:.4f}")
    print(f"    added resistance                 = {surf_res:.4f}")
    print(f"    travel time in waves             = {surf_time:.4f}")
    print(f"  [wave only] wave geodesic travel time   = {wave_time:.4f}")
    print(f"    ground length                    = {wave_ground:.4f}")
    print(f"    added resistance                 = {wave_res:.4f}")
    print(f"  -> wrong-metric time loss          = {100*time_penalty:+.2f}%")
    print(f"  -> added-resistance reduction      = {100*res_reduction:.2f}%")
    print(f"  -> ground-length premium (tacking) = {100*len_penalty:+.2f}%")
    print(f"  [with drift] Randers (current) cost       = {c_cur:.4f}")
    print(f"    wave_current (eps0=0) cost              = {c_wc0:.4f}  (reduces to Randers)")
    print(f"    wave_current (eps0={EPS0:.2f}) cost     = {c_wc:.4f}")
    print(f"    Randers path under full Finsler         = {cur_under_wc:.4f}")
    print(f"  -> Randers wrong-metric time loss         = {100*drift_penalty:+.2f}%")
    print(f"  severity sweep eps0={eps_grid.tolist()}")
    print(f"    wrong-metric loss (%) = {np.round(100*penalties, 3).tolist()}")


def _render_figure(sc_wave, p_surf, p_wave, eps_grid, penalties):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    # (a) surface elevation with the two geodesics overlaid.
    ax = axes[0]
    im = ax.pcolormesh(sc_wave.Xg, sc_wave.Yg, sc_wave.h, shading="auto",
                       cmap=E.CMAP_ELEV, norm=E._elev_norm(sc_wave.h))
    ax.plot([S[0], G[0]], [S[1], G[1]], "--", color="0.35", lw=1.0, alpha=0.8,
            label="straight line", zorder=2)
    ax.plot(p_surf[:, 0], p_surf[:, 1], color=C_ARC, lw=2.2, ls="--",
            label="arc-length geodesic", zorder=4)
    ax.plot(p_wave[:, 0], p_wave[:, 1], color=C_WAV, lw=2.6,
            label="wave geodesic", zorder=5)
    E._plot_start_goal(ax)
    E._format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7)
    E._panel_label(ax, "a")

    # (b) wrong-metric penalty vs sea severity.
    ax = axes[1]
    loss = 100 * penalties
    ax.fill_between(eps_grid, loss, 0.0, color=C_WAV, alpha=0.15, zorder=1)
    ax.axhline(0.0, color="0.4", lw=1.0, ls="--", zorder=2)
    ax.plot(eps_grid, loss, "-o", color=C_WAV, lw=2.0, ms=5, mfc="white",
            mec=C_WAV, mew=1.5, zorder=3)
    ax.set_xlabel("peak head-sea slowdown $\\varepsilon_0$")
    ax.set_ylabel("wrong-metric time loss (%)")
    ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    E._format_axis(ax)
    E._panel_label(ax, "b")

    E._save_fig(fig, "fig_wave_metrics")


if __name__ == "__main__":
    main()
