"""Generate ``fig_real_current.*`` -- the time-optimal path over a REAL ocean-current field.

This figure reads the extracted Gulf Stream snapshot (``data/real_current_gulfstream.npz``,
written by ``real_current.py`` from the AC-PIDNN global product) and renders two panels:

  (a) the raw snapshot in geographic coordinates, showing the western boundary
      current that the paper's synthetic ``current`` scenario emulates;
  (b) the same field resampled onto the paper's 10 x 10 km planning domain, with
      the direct geodesic overlaid to show the planner exploiting the real drift.

The field is REAL observed/modelled data. The only transforms applied are a
non-dimensionalisation by the reference USV speed ``U_ship`` and a linear
resampling onto the domain grid; no synthetic field is substituted. Provenance is
declared in the note printed to stdout.

Run from the ``simulation/`` directory:

    python fig_real_current.py

Outputs
-------
- ``figures/fig_real_current.svg`` : editable vector source.
- ``figures/fig_real_current.pdf`` : publication figure (matplotlib, Agg).
- ``figures/fig_real_current.png`` : 600 dpi preview.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments as E
from real_current import RealCurrentScenario, U_SHIP
from solvers import direct_geodesic, straight_line_cost

NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "real_current_gulfstream.npz")

SPEED_CMAP = "viridis"


def _speed(u, v):
    return np.sqrt(u * u + v * v)


def _panel_a(ax, lon, lat, u_ms, v_ms, vmax):
    """Geographic snapshot of the Gulf Stream surface current (real data)."""
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")
    spd = _speed(u_ms, v_ms)
    im = ax.pcolormesh(lon2d, lat2d, spd, shading="auto", cmap=SPEED_CMAP,
                       vmin=0.0, vmax=vmax)
    ax.quiver(lon2d[::4, ::4], lat2d[::4, ::4],
              u_ms[::4, ::4], v_ms[::4, ::4],
              color="white", alpha=0.85, scale=12.0, width=0.0035,
              headwidth=3.4, headlength=4.6, headaxislength=3.8, zorder=4)
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    return im


def _panel_b(ax, vx, vy, path, vmax):
    """Planning domain with the real current and the direct geodesic."""
    xs = np.linspace(0.0, E.L, vx.shape[0])
    ys = np.linspace(0.0, E.L, vy.shape[0])
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    spd = _speed(vx, vy) * U_SHIP          # back to m/s for a physical label
    im = ax.pcolormesh(Xg, Yg, spd, shading="auto", cmap=SPEED_CMAP, vmin=0.0,
                       vmax=vmax)
    ax.quiver(Xg[::8, ::8], Yg[::8, ::8], vx[::8, ::8], vy[::8, ::8],
              color="white", alpha=0.85, scale=1.1, width=0.0035,
              headwidth=3.4, headlength=4.6, headaxislength=3.8, zorder=4)
    ax.plot([E.START[0], E.GOAL[0]], [E.START[1], E.GOAL[1]], "--", color="0.35",
            lw=1.0, alpha=0.9, label="straight line", zorder=2)
    # A saturated blue (caption: "direct geodesic (blue)") is used instead of the
    # pale "combined" peach so the line stays legible both on the viridis field
    # and inside the framed legend.
    ax.plot(path[:, 0], path[:, 1], color="#1f77b4", lw=2.6,
            label="direct geodesic", zorder=5)
    E._plot_start_goal(ax, textcolor="white")
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    # Externalise the legend below the axes so it does not occlude the field.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    return im


def main():
    data = np.load(NPZ)
    vx, vy = data["vx"], data["vy"]
    lon, lat = data["lon"], data["lat"]
    u_ms, v_ms = data["u_ms"], data["v_ms"]

    sc = RealCurrentScenario(vx, vy)
    c_no_current = float(np.hypot(*(E.GOAL - E.START)))   # flat baseline, no current
    c_straight = straight_line_cost(E.START, E.GOAL, sc)  # straight line through current
    c_dir, path = direct_geodesic(E.START, E.GOAL, sc, M=32, n_restarts=6, seed=0)
    peak = float(_speed(vx, vy).max())
    peak_ms = float(_speed(u_ms, v_ms).max())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)
    im_a = _panel_a(axes[0], lon, lat, u_ms, v_ms, vmax=peak_ms)
    im_b = _panel_b(axes[1], vx, vy, path, vmax=peak_ms)

    for ax in axes:
        E._format_axis(ax)
    E._panel_label(axes[0], "a")
    E._panel_label(axes[1], "b")

    # One shared colourbar: both panels plot the same (m/s) quantity on the
    # same 0..peak_ms scale, so a single bar avoids a redundant second axis.
    fig.colorbar(im_a, ax=axes, shrink=0.9, pad=0.02,
                 label="surface current speed (m/s)")

    E._save_fig(fig, "fig_real_current")

    # Provenance / result note (declares the data are real and where they came from).
    print("fig_real_current.py -- real-data validation figure")
    print("  source : AC-PIDNN daily sea-surface current, 2023-01-08")
    print("           Cui and Cai, University of Macau; DOI 10.5281/zenodo.19966716")
    print("  box    : lon [-80, -60], lat [30, 42]  (0.25 deg grid)")
    print("  speed  : max %.3f m/s" % peak_ms)
    print("  peak dimensionless drift |v| = %.3f  (U_ship = %.1f m/s)" % (peak, U_SHIP))
    print("  cost: no-current straight line = %.4f" % c_no_current)
    print("        straight line through current = %.4f" % c_straight)
    print("        direct geodesic (real current) = %.4f" % c_dir)
    print("  time reduction vs no current = %+.2f%%"
          % (100.0 * (1.0 - c_dir / c_no_current)))
    print("  outputs : figures/fig_real_current.{svg,pdf,png}")


if __name__ == "__main__":
    main()
