"""P4 experiment -- real wave + current dual-field validation.

The manuscript's real-data section currently validates the planner on a real
ocean-current snapshot only (AC-PIDNN, Gulf Stream, 2023-01-08).  This
experiment extends that validation to the *heading-dependent* wave metric of
P1 by pairing the real current with a real wave field, taken from the same
region and date.

Data provenance (real, do not re-fabricate):

  current : AC-PIDNN daily sea-surface current, 2023-01-08 (Cui & Cai,
            U. Macau, DOI 10.5281/zenodo.19966716), resampled onto the 10 km
            domain and non-dimensionalised by U_ship = 4 m/s (peak |v| ~ 0.40).
  waves   : WW3 global wave model (PacIOOS / U. Hawaii), 2023-01-08T00:00Z,
            Gulf Stream box lon 280-300 E, lat 30-42 N, 0.5 deg grid:
              Hs  = 1.90 m (mean), 0.18-3.04 m
              Tdir= 261.5 deg (mean, waves *from* WSW)  ->  propagate *to* 81.5
              Tper= 7.59 s  (deep-water wavelength ~ 90 m, steepness ~ 0.021)

Two honest observations shape the experiment:

1. The wave field is *spatially uniform* at the planning scale: 0.5 deg ~ 55 km
   is far coarser than the 10 km domain, so the wave direction and severity are
   single constants over the domain.  The wave metric therefore contributes a
   *uniform heading-dependent speed*, not a spatial routing effect -- the routing
   is driven entirely by the real current.
2. The wave direction is REAL and physically meaningful: waves propagate ENE
   (81.5 deg oceanographic, i.e. 8.5 deg in the metric's atan2 convention), so
   travel east is *following-sea* (fast) and travel west is *head-sea* (slow),
   reinforcing the eastward Gulf Stream drift.

The severity amplitude ``eps0`` is the one quantity the wave data cannot supply
(it is the vessel's seakeeping speed-loss response).  It is fixed at the paper's
synthetic value ``EPS0 = 0.30`` and swept to show the result is robust.
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
from metrics import Scenario
from solvers import direct_geodesic

NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "real_current_gulfstream.npz")

# --- real WW3 wave statistics (Gulf Stream box, 2023-01-08T00:00Z) ----------
PSI_OCEAN_TO = 81.5          # propagation direction, deg clockwise from north
PSI_OCEAN_FROM = 261.5       # the same direction, "from" (meteorological) form
HS_MEAN = 1.90               # significant wave height (m)
TPER_MEAN = 7.59             # peak period (s)
# Metric convention: psi = atan2(direction_y, direction_x), CCW from +x (east).
PSI_MATH = np.deg2rad(90.0 - PSI_OCEAN_TO)

EPS0 = 0.30                  # nominal peak head-sea slowdown (matches metrics.EPS0)


def _build_scenario(mode, vx, vy, psi=0.0, eps=0.0):
    sc = Scenario(L=E.L, N=257, mode=mode)
    sc._vx = vx
    sc._vy = vy
    if mode == "wave_current":
        sc._psi = np.full_like(vx, psi)
        sc._eps = np.full_like(vx, eps)
    return sc


def main():
    data = np.load(NPZ)
    vx, vy = data["vx"], data["vy"]
    peak = float(np.hypot(vx, vy).max())

    sc_cur = _build_scenario("current", vx, vy)
    c_cur, p_cur = direct_geodesic(E.START, E.GOAL, sc_cur, M=32, n_restarts=6, seed=0)

    sc_wc = _build_scenario("wave_current", vx, vy, PSI_MATH, EPS0)
    c_wc, p_wc = direct_geodesic(E.START, E.GOAL, sc_wc, M=32, n_restarts=6, seed=0)

    # eps0 -> 0 reduction check: wave_current must collapse onto `current`.
    sc_wc0 = _build_scenario("wave_current", vx, vy, PSI_MATH, 0.0)
    c_wc0, _ = direct_geodesic(E.START, E.GOAL, sc_wc0, M=32, n_restarts=6, seed=0)
    red_err = (c_wc0 - c_cur) / c_cur   # ~1e-4 = solver noise (real field is irregular)

    # Severity sweep: how the wave-added time grows with eps0 (real psi fixed).
    eps_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
    c_sweep = np.zeros_like(eps_grid)
    for k, e0 in enumerate(eps_grid):
        sc = _build_scenario("wave_current", vx, vy, PSI_MATH, float(e0))
        c_sweep[k], _ = direct_geodesic(E.START, E.GOAL, sc, M=32, n_restarts=6, seed=0)

    c_no_current = float(np.hypot(*(E.GOAL - E.START)))
    results = {
        "current": {
            "source": "AC-PIDNN DOI 10.5281/zenodo.19966716",
            "peak_drift": peak, "cost": c_cur, "no_current_baseline": c_no_current,
        },
        "waves": {
            "source": "WW3 (PacIOOS) 2023-01-08T00:00Z, lon 280-300E lat 30-42N",
            "Hs_mean_m": HS_MEAN, "Tper_mean_s": TPER_MEAN,
            "dir_from_deg": PSI_OCEAN_FROM, "dir_to_deg": PSI_OCEAN_TO,
            "psi_math_deg": float(np.rad2deg(PSI_MATH)),
            "spatial_scale": "uniform over 10 km domain (0.5 deg source grid)",
        },
        "wave_current": {
            "eps0": EPS0, "cost": c_wc,
            "eps0_zero_cost": c_wc0, "current_cost": c_cur,
            "eps0_zero_reduction_error": red_err,
            "wave_added_time": (c_wc - c_cur) / c_cur,
        },
        "severity_sweep": {"eps0": eps_grid.tolist(),
                           "wave_current_cost": c_sweep.tolist()},
    }
    with open("results_real_wave_current.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _render_figure(vx, vy, p_cur, p_wc, eps_grid, c_sweep, c_cur)

    print("P4 -- real wave + current dual-field validation")
    print(f"  real current: peak |v| = {peak:.3f}  (weak drift < 1)")
    print(f"  real waves  : Hs={HS_MEAN:.2f} m, Tper={TPER_MEAN:.2f} s, "
          f"dir from {PSI_OCEAN_FROM:.1f} deg (propagates to {PSI_OCEAN_TO:.1f})")
    print(f"                psi (atan2) = {np.rad2deg(PSI_MATH):.1f} deg")
    print(f"  no-current straight-line baseline = {c_no_current:.4f}")
    print(f"  current-only (Randers) cost       = {c_cur:.4f}")
    print(f"  wave_current (eps0=0) cost        = {c_wc0:.4f}  "
          f"(reduces to current, rel err {100*red_err:+.2f}%)")
    print(f"  wave_current (eps0={EPS0:.2f}) cost     = {c_wc:.4f}")
    print(f"  wave-added time (vs current-only) = {100*(c_wc-c_cur)/c_cur:+.2f}%")
    print(f"  severity sweep eps0={eps_grid.tolist()}")
    print(f"    wave_current cost = {np.round(c_sweep, 4).tolist()}")


def _render_figure(vx, vy, p_cur, p_wc, eps_grid, c_sweep, c_cur):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    # (a) real current field + current-only vs wave-current geodesics.
    ax = axes[0]
    xs = np.linspace(0.0, E.L, vx.shape[0])
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    spd = np.hypot(vx, vy)
    im = ax.pcolormesh(Xg, Yg, spd, shading="auto", cmap="viridis")
    ax.plot(p_cur[:, 0], p_cur[:, 1], color="#5E6AD2", lw=2.4, ls="--",
            label="current only", zorder=4)
    ax.plot(p_wc[:, 0], p_wc[:, 1], color="#1F9D55", lw=2.6,
            label="wave + current", zorder=5)
    ax.plot([E.START[0], E.GOAL[0]], [E.START[1], E.GOAL[1]], "--", color="0.35",
            lw=1.0, alpha=0.8, label="straight line", zorder=2)
    E._plot_start_goal(ax, textcolor="white")
    E._format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7)
    E._panel_label(ax, "a")

    # (b) severity sweep of the wave-added cost (real psi fixed).
    ax = axes[1]
    ax.plot(eps_grid, c_sweep, "o-", color="#1F9D55", lw=1.6, ms=5, mfc="white",
            mec="#1F9D55", mew=1.3, zorder=3)
    ax.axhline(c_cur, color="#5E6AD2", lw=1.4, ls="--", label="current only",
               zorder=1)
    ax.set_xlabel("peak head-sea slowdown $\\varepsilon_0$")
    ax.set_ylabel("travel time")
    ax.grid(True, color="0.85", lw=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=7)
    E._format_axis(ax)
    E._panel_label(ax, "b")

    E._save_fig(fig, "fig_real_wave_current")


if __name__ == "__main__":
    main()
