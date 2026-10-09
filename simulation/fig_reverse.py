"""Generate ``fig_reverse.*`` -- optimal paths for the reversed start/goal scenario.

Reads the ``reverse_benchmark`` block from ``results.json`` (produced by
``exp_reverse`` in ``experiments.py``) and renders the arrival-time field plus the
flat / surface / current / combined optimal geodesics for (9,9) -> (1,1): the same
sea surface and current field as the primary scenario with only the endpoints
swapped. This is the second experimental scenario introduced in response to the
review, and it makes the Randers metric's asymmetry visible directly.

Run from the ``simulation/`` directory:

    python fig_reverse.py

Outputs
-------
- ``figures/fig_reverse.pdf`` : publication figure (matplotlib, Agg backend).
- ``figures/fig_reverse.svg`` : editable vector source.
- ``figures/fig_reverse.png`` : 600 dpi preview.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import experiments as E
from solvers import grid_dijkstra


def _plot_reverse_start_goal(ax):
    """Mark start=(9,9) and goal=(1,1) with annotations swapped vs the primary."""
    ax.scatter(*E.GOAL, marker="o", s=90, facecolor="white",
               edgecolor=E.C_START, linewidth=1.2, zorder=6)
    ax.scatter(*E.START, marker="*", s=220, color=E.C_GOAL,
               edgecolor=E.C_START, linewidth=0.5, zorder=6)
    ax.annotate("start", E.GOAL + np.array([0.25, 0.30]), fontsize=8,
                ha="left", va="bottom", zorder=7)
    ax.annotate("goal", E.START + np.array([-0.25, -0.30]), fontsize=8,
                ha="right", va="top", zorder=7)


def main():
    with open("results.json", encoding="utf-8") as f:
        results = json.load(f)
    reverse = results["reverse_benchmark"]

    sc = E.make_scenario("combined")
    # Dijkstra must propagate from the *start* (9,9): the Randers metric is
    # asymmetric, so the arrival-time field is the (9,9) -> x field, not the
    # time-reversal of the primary (1,1) -> x field.
    _, _, dist, xs, ys = grid_dijkstra(sc, E.GOAL, E.START, N_d=E.N_D)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cs = ax.contourf(xs, ys, dist.T, levels=40, cmap=E.CMAP_TIME)
    fig.colorbar(cs, ax=ax, label="arrival time (dimensionless)",
                 shrink=0.82, pad=0.02)

    ax.plot([E.GOAL[0], E.START[0]], [E.GOAL[1], E.START[1]], "--", color="0.35",
            lw=1.0, alpha=0.8, label="straight line", zorder=2)
    LINE_STYLES = {"flat": "--", "surface": "-.", "current": "-", "combined": "-"}
    for mode in ["flat", "surface", "current", "combined"]:
        p = np.array(reverse[mode]["path_direct"])
        lw = 2.6 if mode == "combined" else 1.7
        z = 4 if mode == "combined" else 3
        ax.plot(p[:, 0], p[:, 1], color=E.PATH_COLORS[mode], lw=lw,
                ls=LINE_STYLES[mode], label=E.PATH_LABELS[mode], zorder=z)
    _plot_reverse_start_goal(ax)
    E._format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    E._save_fig(fig, "fig_reverse")


if __name__ == "__main__":
    main()
