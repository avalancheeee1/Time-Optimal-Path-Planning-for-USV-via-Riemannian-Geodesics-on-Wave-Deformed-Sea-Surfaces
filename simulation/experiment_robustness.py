"""P3 experiment -- robustness across current geometry and start/goal pair.

The primary benchmark uses a single Rankine-like eddy and the (1,1)->(9,9)
pair.  To show the conclusions are not tuned to that one scenario, this
experiment re-runs the four-mode benchmark on a *different* current geometry
(a meandering eastward jet, Gulf-Stream-like) and a *different* start/goal pair
that crosses the jet.  The same three numbers are reported for every mode:

* ``straight_line_cost``   -- the naive Euclidean reference,
* ``dijkstra_cost``        -- the 8-connected grid search (discrete upper bound),
* ``direct_geodesic_cost`` -- the proposed continuous method,

and the direct solver's time saving over straight-line routing is compared with
the primary scenario.  Agreement of the qualitative behaviour (direct beats
straight, direct <= Dijkstra) across both geometries and both pairs is the
robustness claim.
"""
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
from solvers import grid_dijkstra, direct_geodesic, straight_line_cost


# --- second current geometry: a meandering eastward jet -----------------------
V_JET = 0.55          # jet core speed
Y0 = 5.0              # mean jet latitude
A_MEANDER = 1.6       # meander amplitude
K_JET = 2 * np.pi / 10.0   # one meander wavelength across the domain
W_JET = 1.1           # cross-jet half-width


def meander_jet(X, Y):
    """A smooth eastward meandering jet: Gaussian cross-jet profile along a
    sinusoidal axis, with a vertical component that follows the meander."""
    yc = Y0 + A_MEANDER * np.sin(K_JET * X)
    d = Y - yc
    f = np.exp(-(d * d) / (W_JET * W_JET))
    vx = V_JET * f
    vy = V_JET * A_MEANDER * K_JET * np.cos(K_JET * X) * f
    return vx, vy


# --- second start/goal pair (crosses the jet NW -> SE) -----------------------
START2 = np.array([1.0, 7.0])
GOAL2 = np.array([9.0, 3.0])


def make_jet_scenario(mode):
    sc = E.make_scenario(mode)
    if mode in ("current", "combined"):
        sc._vx, sc._vy = meander_jet(sc.Xg, sc.Yg)
    return sc


def _benchmark(start, goal):
    modes = ["flat", "surface", "current", "combined"]
    rows = {}
    for mode in modes:
        sc = make_jet_scenario(mode)
        c_dij, path_dij, *_ = grid_dijkstra(sc, start, goal, N_d=E.N_D)
        c_dir, path_dir = direct_geodesic(start, goal, sc, M=32, n_restarts=6,
                                          seed=E.SEED)
        c_str = straight_line_cost(start, goal, sc)
        rows[mode] = {
            "dijkstra_cost": c_dij,
            "direct_geodesic_cost": c_dir,
            "straight_line_cost": c_str,
            "path_length_ground": E.path_length_ground(path_dir),
            "lateral_deviation": E.lateral_deviation(path_dir, start, goal),
            "weak_drift_max": E.weak_drift_max(sc),
            "path_direct": path_dir.tolist(),
            "path_dijkstra": path_dij.tolist(),
        }
    c_str_flat = rows["flat"]["straight_line_cost"]
    for mode in rows:
        rows[mode]["time_reduction_vs_straight"] = (
            (c_str_flat - rows[mode]["direct_geodesic_cost"]) / c_str_flat
        )
    return rows


def main():
    rows = _benchmark(START2, GOAL2)

    results = {"start": START2.tolist(), "goal": GOAL2.tolist(),
               "geometry": "meandering_jet",
               "jet_params": {"V": V_JET, "y0": Y0, "A": A_MEANDER,
                              "k": K_JET, "W": W_JET},
               "benchmark": rows}
    with open("results_robustness.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _render_figure(rows)

    print("P3 -- robustness: meandering jet + second start/goal pair")
    print(f"  start={START2.tolist()} goal={GOAL2.tolist()}")
    print(f"  max |v|_g over modes = "
          f"{max(rows[m]['weak_drift_max'] for m in rows):.4f}  (< 1 required)")
    for m in rows:
        r = rows[m]
        print(f"  {m:9s} straight={r['straight_line_cost']:.4f}  "
              f"dijkstra={r['dijkstra_cost']:.4f}  direct={r['direct_geodesic_cost']:.4f}  "
              f"save={100*r['time_reduction_vs_straight']:+.2f}%")
    # direct <= dijkstra (continuous vs discrete) in every current mode
    ok = all(rows[m]["direct_geodesic_cost"] <= rows[m]["dijkstra_cost"] + 1e-9
             for m in ("current", "combined"))
    print(f"  -> direct <= dijkstra in current/combined: {ok}")


def _render_figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    sc = make_jet_scenario("combined")
    ax = axes[0]
    im = ax.pcolormesh(sc.Xg, sc.Yg, sc._vx, shading="auto", cmap=E.CMAP_ELEV)
    p = np.array(rows["combined"]["path_direct"])
    ax.plot(p[:, 0], p[:, 1], color="#1F9D55", lw=2.6, label="combined geodesic",
            zorder=5)
    ax.plot([START2[0], GOAL2[0]], [START2[1], GOAL2[1]], "--", color="0.35",
            lw=1.0, alpha=0.8, label="straight line", zorder=2)
    E._plot_start_goal(ax)
    E._format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=7)
    E._panel_label(ax, "a")

    ax = axes[1]
    modes = ["flat", "surface", "current", "combined"]
    labels = ["flat", "surface", "current", "combined"]
    saves = [100 * rows[m]["time_reduction_vs_straight"] for m in modes]
    bars = ax.bar(labels, saves, color=["#9AA0B4", "#5E6AD2", "#C8683A", "#1F9D55"],
                  width=0.62, zorder=3)
    ax.axhline(0.0, color="0.35", lw=0.8, zorder=1)
    ax.set_ylabel("time saving vs straight line (%)")
    ax.grid(True, axis="y", color="0.85", lw=0.5, zorder=0)
    E._format_axis(ax)
    E._panel_label(ax, "b")

    E._save_fig(fig, "fig_robustness")


if __name__ == "__main__":
    main()
