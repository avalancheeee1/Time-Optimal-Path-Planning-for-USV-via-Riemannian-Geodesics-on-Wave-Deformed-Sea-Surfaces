"""Reproducible experiments for the USV Riemannian-geodesic path-planning paper.

Every number and figure in the manuscript is produced by running this script
(deterministic seeds, fixed parameters). Run from the ``simulation/`` directory:

    python experiments.py

Outputs
-------
- ``figures/*.pdf``   : publication figures (matplotlib, Agg backend).
- ``results.json``    : machine-readable results for the paper's tables.
- ``results.md``      : human-readable summary of the same numbers.
"""

import json
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from metrics import Scenario, surface_metric, current_field, surface_height, surface_field_spectral
from solvers import (
    grid_dijkstra,
    direct_geodesic,
    straight_line_cost,
    polyline_cost,
)
from fmm import fmm_randers, fmm_randers_lbr


# --------------------------------------------------------------------------- #
# Global setup
# --------------------------------------------------------------------------- #
L = 10.0          # domain [0, L] x [0, L] (dimensionless; paper maps to km)
N_FIELD = 257     # field interpolation grid
N_D = 193         # Dijkstra grid resolution
SEED = 0
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Base scenario parameters (deterministic).
A_DEFAULT = (0.5, 0.3)
LAM_DEFAULT = (10.0, 5.0)
PHI_DEFAULT = 0.5
U_DEFAULT = 0.40
U0_DEFAULT = 0.15
SIGMA_DEFAULT = 2.5
XC_DEFAULT = 5.0
YC_DEFAULT = 5.0

START = np.array([1.0, 1.0])
GOAL = np.array([9.0, 9.0])


# --------------------------------------------------------------------------- #
# Publication figure style (Nature-family conventions: sans-serif Arial)
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans",
                        "sans-serif"],
    "font.size": 7,
    "mathtext.fontset": "dejavusans",   # sans-serif math to match Arial
    "axes.linewidth": 0.8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",            # keep text as editable <text> nodes
    "pdf.fonttype": 42,                # embed TrueType fonts (publisher-safe)
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Warm sequential ramp sampled from 'magma', aligned with the Fig. 5 regime map
# (CMAP_DOM).  Monotonic luminance keeps the four bars separable in grayscale
# (IEEE black-and-white readability); green/red remain reserved for delta cues.
C_FLAT     = "#251255"   # magma 0.15 -- dark indigo (Euclidean baseline)
C_SURFACE  = "#A1307E"   # magma 0.45 -- magenta (surface only)
C_CURRENT  = "#F9795D"   # magma 0.72 -- coral (current only)
C_COMBINED = "#FED89A"   # magma 0.92 -- pale peach (combined)
PATH_COLORS = {"flat": C_FLAT, "surface": C_SURFACE,
               "current": C_CURRENT, "combined": C_COMBINED}
PATH_LABELS = {"flat": "flat (Euclidean)", "surface": "surface only",
               "current": "current only", "combined": "combined"}

C_START = "#272727"      # start marker edge (neutral black)
C_GOAL  = "#B64342"      # goal marker (red strong)
C_GAIN  = "#2E9E44"      # time saved (directional cue)
C_COST  = "#E53935"      # time added (directional cue)

CMAP_ELEV = "RdBu_r"     # signed surface elevation (blue troughs -> red crests)
CMAP_TIME = "magma"      # arrival-time field (sequential)
CMAP_DOM  = "magma"      # log dominance ratio (sequential; current always dominates here)
CMAP_SEQ  = "viridis"    # lateral deviation (sequential)


def _format_axis(ax):
    """Remove top/right spines and set consistent outward ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="out", length=3.0, width=0.8)


def _panel_label(ax, label):
    """Bold lowercase panel label at the top-left corner (Nature style)."""
    ax.text(-0.06, 1.02, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def _plot_start_goal(ax):
    """Mark and annotate the start and goal waypoints."""
    ax.scatter(*START, marker="o", s=90, facecolor="white",
               edgecolor=C_START, linewidth=1.2, zorder=6)
    ax.scatter(*GOAL, marker="*", s=220, color=C_GOAL,
               edgecolor=C_START, linewidth=0.5, zorder=6)
    ax.annotate("start", START + np.array([-0.25, -0.30]), fontsize=7,
                ha="right", va="top", zorder=7)
    ax.annotate("goal", GOAL + np.array([0.25, 0.30]), fontsize=7,
                ha="left", va="bottom", zorder=7)


def _elev_norm(h):
    """Two-slope norm centred at zero for signed elevation fields."""
    vmax = float(np.abs(h).max())
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _save_fig(fig, name):
    """Save as SVG (primary, editable) + PDF (LaTeX) + PNG (600 dpi preview)."""
    for ext in ("svg", "pdf", "png"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), **kw)
    plt.close(fig)


def make_scenario(mode: str, A=A_DEFAULT, U=U_DEFAULT, U0=U0_DEFAULT):
    return Scenario(
        L=L, N=N_FIELD, mode=mode, A=A, lam=LAM_DEFAULT, phi=PHI_DEFAULT,
        U=U, U0=U0, sigma=SIGMA_DEFAULT, xc=XC_DEFAULT, yc=YC_DEFAULT,
    )


# --------------------------------------------------------------------------- #
# Small geometric helpers
# --------------------------------------------------------------------------- #
def path_length_ground(path: np.ndarray) -> float:
    """Euclidean length of the planar (ground-projected) path."""
    d = np.diff(path, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def lateral_deviation(path: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    """Max perpendicular distance of a path from the straight start->goal line."""
    a = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    n = np.hypot(a[0], a[1])
    if n < 1e-12:
        return 0.0
    u = a / n                                   # unit direction
    # perpendicular distance of each path point to the infinite line
    rel = np.asarray(path, dtype=float) - np.asarray(start, dtype=float)
    perp = rel[:, 0] * (-u[1]) + rel[:, 1] * u[0]
    return float(np.max(np.abs(perp)))


def weak_drift_max(sc: Scenario) -> float:
    """max |v|_g over the field grid (must be < 1 for the Zermelo/Randers model)."""
    gxx, gxy, gyy, *_ = surface_metric(sc._hx, sc._hy)
    vx, vy = sc._vx, sc._vy
    gv = gxx * vx * vx + 2 * gxy * vx * vy + gyy * vy * vy
    return float(np.sqrt(gv).max())


# --------------------------------------------------------------------------- #
# Experiment 1 -- analytic correctness (reproduces test_sanity.py numbers)
# --------------------------------------------------------------------------- #
def exp_correctness(results: dict):
    euclid = float(np.hypot(*(GOAL - START)))

    # flat -> Euclidean
    sc_flat = make_scenario("flat")
    c_dij, *_ = grid_dijkstra(sc_flat, START, GOAL, N_d=N_D)
    c_dir, _ = direct_geodesic(START, GOAL, sc_flat, M=32, n_restarts=6, seed=SEED)
    c_str = straight_line_cost(START, GOAL, sc_flat)
    c_fmm, *_ = fmm_randers(sc_flat, START, GOAL, N_d=N_D, stencil=8)

    # constant current -> straight line in ground frame (v = (0.4, 0))
    sc_const = make_scenario("current", U=0.0, U0=0.40)
    from metrics import randers_flat_cost_vec
    dvec = (GOAL - START)[None, :]
    ref_const = float(randers_flat_cost_vec(dvec, np.array([0.40]), np.array([0.0]))[0])
    c_dij_c, *_ = grid_dijkstra(sc_const, START, GOAL, N_d=N_D)
    c_dir_c, _ = direct_geodesic(START, GOAL, sc_const, M=32, n_restarts=6, seed=SEED)
    c_str_c = straight_line_cost(START, GOAL, sc_const)
    c_fmm_c, *_ = fmm_randers(sc_const, START, GOAL, N_d=N_D, stencil=8)

    # Lattice-basis-reduction FMM (Mirebeau 2014).  For the near-isotropic flat
    # and constant-current metrics the Selling decomposition collapses to the
    # cardinal pair, so it reduces to the 4-neighbour Godunov scheme (== fmm(4)).
    c_lbr, *_ = fmm_randers_lbr(sc_flat, START, GOAL, N_d=N_D)
    c_lbr_c, *_ = fmm_randers_lbr(sc_const, START, GOAL, N_d=N_D)

    results["correctness"] = {
        "euclidean_ground_truth": euclid,
        "flat": {"dijkstra": c_dij, "direct_geodesic": c_dir,
                 "straight_line": c_str, "fmm": c_fmm, "lbr": c_lbr},
        "constant_current": {
            "analytic_reference": ref_const,
            "dijkstra": c_dij_c,
            "direct_geodesic": c_dir_c,
            "straight_line": c_str_c,
            "fmm": c_fmm_c,
            "lbr": c_lbr_c,
        },
    }


# --------------------------------------------------------------------------- #
# Experiment 2 -- four-mode benchmark on the primary scenario
# --------------------------------------------------------------------------- #
def _run_benchmark(start, goal):
    """Four-mode benchmark for an arbitrary (start, goal) pair.

    Because the Randers metric is asymmetric, ``_run_benchmark(goal, start)`` is
    *not* the time-reversal of ``_run_benchmark(start, goal)``: the two optima are
    distinct geodesics, and both are computed independently here.
    """
    modes = ["flat", "surface", "current", "combined"]
    rows = {}
    for mode in modes:
        sc = make_scenario(mode)
        c_dij, path_dij, dist, xs, ys = grid_dijkstra(sc, start, goal, N_d=N_D)
        c_dir, path_dir = direct_geodesic(start, goal, sc, M=32, n_restarts=6, seed=SEED)
        c_str = straight_line_cost(start, goal, sc)
        rows[mode] = {
            "dijkstra_cost": c_dij,
            "direct_geodesic_cost": c_dir,
            "straight_line_cost": c_str,
            "path_length_ground": path_length_ground(path_dir),
            "lateral_deviation": lateral_deviation(path_dir, start, goal),
            "weak_drift_max": weak_drift_max(sc),
            "path_direct": path_dir.tolist(),
            "path_dijkstra": path_dij.tolist(),
        }
    # time savings relative to the straight-line (Euclidean) reference
    c_str_flat = rows["flat"]["straight_line_cost"]
    for mode in rows:
        rows[mode]["time_reduction_vs_straight"] = (
            (c_str_flat - rows[mode]["direct_geodesic_cost"]) / c_str_flat
        )
    return rows


def exp_benchmark(results: dict):
    rows = _run_benchmark(START, GOAL)
    results["benchmark"] = rows
    return rows


# --------------------------------------------------------------------------- #
# Experiment 8 -- reverse-scenario benchmark, (9,9) -> (1,1)
# --------------------------------------------------------------------------- #
def exp_reverse(results: dict):
    """Four-way benchmark with the start and goal reversed.

    Since the Randers metric is asymmetric, the optimal path from (9,9) to (1,1)
    is a distinct geodesic from the time-reversal of the (1,1)->(9,9) path; this
    second scenario tests whether the paper's qualitative conclusions survive a
    direction reversal (up- versus down-current routing)."""
    rows = _run_benchmark(GOAL, START)
    results["reverse_benchmark"] = rows
    return rows


# --------------------------------------------------------------------------- #
# Experiment 3 -- regime / ablation sweep over (wave amplitude A, current U)
# --------------------------------------------------------------------------- #
def exp_regime(results: dict):
    A_grid = np.linspace(0.1, 1.0, 6)
    U_grid = np.linspace(0.05, 0.50, 6)
    nA, nU = A_grid.size, U_grid.size

    cost_flat = straight_line_cost(START, GOAL, make_scenario("flat"))

    cost_combined = np.zeros((nA, nU))
    cost_surface = np.zeros((nA, nU))
    cost_current = np.zeros((nA, nU))
    dev_combined = np.zeros((nA, nU))

    for i, A in enumerate(A_grid):
        for j, U in enumerate(U_grid):
            sc_c = make_scenario("combined", A=(A, 0.6 * A), U=U)
            sc_s = make_scenario("surface", A=(A, 0.6 * A))
            sc_u = make_scenario("current", U=U)
            c_c, p_c = direct_geodesic(START, GOAL, sc_c, M=32, n_restarts=3, seed=SEED)
            c_s, _ = direct_geodesic(START, GOAL, sc_s, M=32, n_restarts=3, seed=SEED)
            c_u, _ = direct_geodesic(START, GOAL, sc_u, M=32, n_restarts=3, seed=SEED)
            cost_combined[i, j] = c_c
            cost_surface[i, j] = c_s
            cost_current[i, j] = c_u
            dev_combined[i, j] = lateral_deviation(p_c, START, GOAL)

    # "added cost" of each term, and the dominance ratio current/surface
    added_surface = (cost_surface - cost_flat) / cost_flat
    added_current = (cost_current - cost_flat) / cost_flat
    added_combined = (cost_combined - cost_flat) / cost_flat
    # non-additivity: does the joint metric exceed the sum of separate effects?
    nonadd = added_combined - (added_surface + added_current)

    results["regime"] = {
        "A_grid": A_grid.tolist(),
        "U_grid": U_grid.tolist(),
        "cost_flat": cost_flat,
        "added_surface_frac": added_surface.tolist(),
        "added_current_frac": added_current.tolist(),
        "added_combined_frac": added_combined.tolist(),
        "nonadditivity_frac": nonadd.tolist(),
        "lateral_deviation_combined": dev_combined.tolist(),
    }
    return {
        "A_grid": A_grid, "U_grid": U_grid,
        "added_surface": added_surface, "added_current": added_current,
        "added_combined": added_combined, "nonadd": nonadd, "dev": dev_combined,
    }


# --------------------------------------------------------------------------- #
# Experiment 4 -- resolution independence of the direct geodesic
# --------------------------------------------------------------------------- #
def exp_convergence(results: dict):
    sc = make_scenario("combined")
    M_grid = [4, 8, 16, 32, 64]
    costs = []
    for M in M_grid:
        c, _ = direct_geodesic(START, GOAL, sc, M=M, n_restarts=6, seed=SEED)
        costs.append(c)
    results["convergence"] = {"M_grid": M_grid, "cost": costs}
    return M_grid, np.array(costs)


# --------------------------------------------------------------------------- #
# Experiment 5 -- multi-seed robustness of the direct solver
# --------------------------------------------------------------------------- #
def exp_robustness(results: dict):
    """Does the multi-restart solver converge to the *same* minimizer across
    independent seeds? Runs the combined scenario for 10 seeds and records the
    spread of the converged cost."""
    sc = make_scenario("combined")
    seeds = list(range(10))
    costs = []
    for s in seeds:
        c, _ = direct_geodesic(START, GOAL, sc, M=32, n_restarts=6, seed=s)
        costs.append(c)
    costs = np.array(costs)
    results["robustness"] = {
        "seed_list": seeds,
        "cost_list": costs.tolist(),
        "min": float(costs.min()),
        "max": float(costs.max()),
        "mean": float(costs.mean()),
        "std": float(costs.std()),
        "spread_rel": float((costs.max() - costs.min()) / costs.min()),
    }
    return costs


# --------------------------------------------------------------------------- #
# Experiment 6 -- computational cost vs the grid eikonal solver
# --------------------------------------------------------------------------- #
def exp_runtime(results: dict):
    """Wall-clock time and attained cost of the direct solver versus the grid
    Dijkstra baseline across resolutions and stencil connectivities. Costs are
    deterministic (fixed seed); timings are single-run wall-clock on this host."""
    sc = make_scenario("combined")

    direct = {}
    for M in [8, 16, 32]:
        t0 = time.perf_counter()
        c, _ = direct_geodesic(START, GOAL, sc, M=M, n_restarts=6, seed=SEED)
        direct[M] = {"cost": c, "time_s": time.perf_counter() - t0}

    grid = {}
    for N_d in [97, 193, 385]:
        t0 = time.perf_counter()
        c, *_ = grid_dijkstra(sc, START, GOAL, N_d=N_d)
        grid[N_d] = {"cost": c, "time_s": time.perf_counter() - t0}

    # Wider stencils (16- / 24-connected) quantify the metrication error: the
    # discrete cost should fall as connectivity grows, approaching the continuous
    # optimum from above but (for a fixed stencil) never reaching it exactly.
    grid_wider = {}
    for conn in (16, 24):
        for N_d in (193, 385):
            t0 = time.perf_counter()
            c, *_ = grid_dijkstra(sc, START, GOAL, N_d=N_d, connectivity=conn)
            grid_wider[f"{conn}_{N_d}"] = {"connectivity": conn, "N": N_d,
                                           "cost": c, "time_s": time.perf_counter() - t0}

    # Upwind fast-marching (FMM) baseline: a first-order Godunov scheme for the
    # anisotropic eikonal F*(x, grad T) = 1.  Unlike fixed-compass Dijkstra it
    # propagates the front over a continuum of headings, so it removes most of the
    # narrow-stencil metrication error at the same neighbour support.
    fmm = {}
    for st in (8, 24):
        for N_d in (97, 193, 385):
            t0 = time.perf_counter()
            c, *_ = fmm_randers(sc, START, GOAL, N_d=N_d, stencil=st)
            fmm[f"{st}_{N_d}"] = {"stencil": st, "N": N_d,
                                  "cost": c, "time_s": time.perf_counter() - t0}

    # Lattice-basis-reduction FMM (Mirebeau 2014): a metric-adapted Selling
    # stencil.  For this near-isotropic scenario the stencil collapses to the
    # cardinal pair (a 4-neighbour scheme), so it carries the same O(h) diagonal
    # bias and does not improve on the fixed 8/24-neighbour FMM.  Reported at two
    # resolutions for completeness.  (The shifted dual metric is non-causal, so the
    # monotone scheme is fast-swept rather than single-pass marched; the timing is
    # for this pure-Python prototype, not the compiled HFM library.)
    lbr = {}
    for N_d in (97, 193):
        t0 = time.perf_counter()
        c, *_ = fmm_randers_lbr(sc, START, GOAL, N_d=N_d)
        lbr[N_d] = {"cost": c, "time_s": time.perf_counter() - t0}

    results["runtime"] = {"direct": direct, "grid": grid,
                          "grid_wider": grid_wider, "fmm": fmm, "lbr": lbr}
    return direct, grid


# --------------------------------------------------------------------------- #
# Experiment 7 -- irregular (Pierson--Moskowitz) sea benchmark
# --------------------------------------------------------------------------- #
def make_spectral_scenario(mode: str, seed: int = 7):
    """Build a scenario whose surface is a spectral realisation and whose current
    is a large-scale gyre, overwriting the analytic fields of ``make_scenario``."""
    sc = make_scenario(mode)
    if mode in ("surface", "combined"):
        h, hx, hy = surface_field_spectral(sc.Xg, sc.Yg, seed=seed, Hs=0.5)
        sc.h = h
        sc._hx = hx
        sc._hy = hy
    if mode in ("current", "combined"):
        # A larger, slower gyre (more realistic large-scale structure).
        vx, vy = current_field(sc.Xg, sc.Yg, U=0.30, U0=0.15, xc=5.0, yc=5.0, sigma=3.5)
        sc._vx = vx
        sc._vy = vy
    return sc


def exp_realistic(results: dict):
    """Four-way benchmark on the irregular sea (spectral surface + gyre current)."""
    modes = ["flat", "surface", "current", "combined"]
    rows = {}
    for mode in modes:
        sc = make_spectral_scenario(mode, seed=7)
        c_dir, path_dir = direct_geodesic(START, GOAL, sc, M=32, n_restarts=6, seed=SEED)
        c_str = straight_line_cost(START, GOAL, sc)
        rows[mode] = {
            "direct_geodesic_cost": c_dir,
            "straight_line_cost": c_str,
            "path_length_ground": path_length_ground(path_dir),
            "lateral_deviation": lateral_deviation(path_dir, START, GOAL),
            "weak_drift_max": weak_drift_max(sc),
            "path_direct": path_dir.tolist(),
        }
    c_str_flat = rows["flat"]["straight_line_cost"]
    for mode in rows:
        rows[mode]["time_reduction_vs_straight"] = (
            (c_str_flat - rows[mode]["direct_geodesic_cost"]) / c_str_flat
        )
    results["realistic_benchmark"] = rows
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_scenario(sc: Scenario):
    """Surface elevation + current vector field."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    h = surface_height(sc.Xg, sc.Yg, A=A_DEFAULT, lam=LAM_DEFAULT, phi=PHI_DEFAULT)
    im = ax.pcolormesh(sc.Xg, sc.Yg, h, shading="auto", cmap=CMAP_ELEV,
                       norm=_elev_norm(h))
    step = 10
    ax.quiver(
        sc.Xg[::step, ::step], sc.Yg[::step, ::step],
        sc._vx[::step, ::step], sc._vy[::step, ::step],
        color="0.18", alpha=0.6, scale=7.0, width=0.0028,
        headwidth=3.2, headlength=4.2, headaxislength=3.6, zorder=4,
    )
    _plot_start_goal(ax)
    _format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, label="surface elevation $h$", shrink=0.82, pad=0.02)
    fig.tight_layout()
    _save_fig(fig, "fig_scenario")


def fig_paths(benchmark: dict, sc: Scenario):
    """Arrival-time field (combined) with optimal paths for all four modes."""
    c_dij, path_dij, dist, xs, ys = grid_dijkstra(sc, START, GOAL, N_d=N_D)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cs = ax.contourf(xs, ys, dist.T, levels=40, cmap=CMAP_TIME)
    cb = fig.colorbar(cs, ax=ax, label="arrival time (dimensionless)",
                      shrink=0.82, pad=0.02)

    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color="0.35", lw=1.0,
            alpha=0.8, label="straight line", zorder=2)
    LINE_STYLES = {"flat": "--", "surface": "-.", "current": "-", "combined": "-"}
    for mode in ["flat", "surface", "current", "combined"]:
        p = np.array(benchmark[mode]["path_direct"])
        lw = 2.6 if mode == "combined" else 1.7
        z = 4 if mode == "combined" else 3
        ls = LINE_STYLES[mode]
        ax.plot(p[:, 0], p[:, 1], color=PATH_COLORS[mode], lw=lw, ls=ls,
                label=PATH_LABELS[mode], zorder=z)
    _plot_start_goal(ax)
    _format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "fig_paths")


def fig_benchmark(benchmark: dict):
    modes = ["flat", "surface", "current", "combined"]
    labels = ["flat\n(Euclidean)", "surface\nonly", "current\nonly",
              "combined\n(wave + current)"]
    c_dir = [benchmark[m]["direct_geodesic_cost"] for m in modes]
    c_str = benchmark["flat"]["straight_line_cost"]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    x = np.arange(len(modes))
    bars = ax.bar(x, c_dir, color=[PATH_COLORS[m] for m in modes], width=0.58,
                  edgecolor="0.25", linewidth=0.6, zorder=3)
    ax.axhline(c_str, color="0.3", ls="--", lw=1.0, zorder=2)
    ax.text(3.44, c_str, "straight-line\nreference", va="bottom", ha="right",
            fontsize=6.5, color="0.3")
    for xi, c in zip(x, c_dir):
        pct = 100 * (c_str - c) / c_str
        ax.text(xi, c + 0.10, f"{c:.2f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
        color = C_COST if pct < -0.05 else (C_GAIN if pct > 0.05 else "0.35")
        ax.text(xi, c + 0.72, f"{pct:+.1f}%", ha="center", va="bottom",
                fontsize=6.5, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("travel time (dimensionless)")
    ax.set_ylim(0, 1.30 * max(c_dir))
    ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
    _format_axis(ax)
    fig.tight_layout()
    _save_fig(fig, "fig_benchmark")


def fig_regime(regime: dict):
    A_grid = regime["A_grid"]
    U_grid = regime["U_grid"]
    # Dominance ratio: |current added cost| / |surface added cost|.  Both are
    # *signed* fractional changes relative to the flat reference -- the current
    # term is negative (it saves time) and the surface term is positive -- so
    # the *magnitudes* are compared.  A value of 0 means the two effects are
    # equal; >0 means the current dominates.  (The old code clipped added_current
    # with ``np.clip(x, 1e-6, None)``, which flattened every negative entry to
    # 1e-6 and yielded a meaningless all-negative, U-independent map.)
    ratio = np.log10(np.abs(regime["added_current"])
                     / np.abs(regime["added_surface"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax = axes[0]
    # Sequential colormap: the dominance ratio is >= 0 across this sweep (the
    # current always dominates), so a diverging map would waste its negative
    # half and collapse the heatmap onto a single dark color.  ``ratio`` and
    # ``dev`` are already (A, U)-indexed, so no transpose is applied.
    cs = ax.contourf(U_grid, A_grid, ratio, levels=30, cmap=CMAP_DOM,
                     vmin=0.0, vmax=2.5)
    fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02,
                 label=r"$\log_{10}$ (current / surface added cost)")
    ax.set_xlabel("current strength $U$")
    ax.set_ylabel("wave amplitude $A$")
    _panel_label(ax, "a")

    ax = axes[1]
    cs2 = ax.contourf(U_grid, A_grid, regime["dev"], levels=30, cmap=CMAP_SEQ)
    fig.colorbar(cs2, ax=ax, shrink=0.85, pad=0.02, label="lateral deviation")
    ax.set_xlabel("current strength $U$")
    ax.set_ylabel("wave amplitude $A$")
    _panel_label(ax, "b")

    for a in axes:
        _format_axis(a)
    fig.tight_layout()
    _save_fig(fig, "fig_regime")


def fig_convergence(M_grid, costs):
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.plot(M_grid, costs, "o-", color=C_COMBINED, lw=1.5, ms=5,
            mfc="white", mec=C_COMBINED, mew=1.2, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(M_grid)
    ax.set_xticklabels([str(int(m)) for m in M_grid])
    ax.minorticks_off()
    ax.set_xlabel("interior points $M$")
    ax.set_ylabel("optimal cost")
    ax.grid(True, which="major", color="0.85", lw=0.5, zorder=0)
    _format_axis(ax)
    fig.tight_layout()
    _save_fig(fig, "fig_convergence")


def fig_realistic(sc: Scenario, realistic: dict):
    """Irregular sea elevation + current field + combined optimal path."""
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    im = ax.pcolormesh(sc.Xg, sc.Yg, sc.h, shading="auto", cmap=CMAP_ELEV,
                       norm=_elev_norm(sc.h))
    step = 10
    ax.quiver(
        sc.Xg[::step, ::step], sc.Yg[::step, ::step],
        sc._vx[::step, ::step], sc._vy[::step, ::step],
        color="0.18", alpha=0.6, scale=5.5, width=0.0028,
        headwidth=3.2, headlength=4.2, headaxislength=3.6, zorder=4,
    )
    p = np.array(realistic["combined"]["path_direct"])
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], "--", color="0.35", lw=1.0,
            alpha=0.8, label="straight line", zorder=2)
    ax.plot(p[:, 0], p[:, 1], color=C_COMBINED, lw=2.6,
            label="combined geodesic", zorder=3)
    _plot_start_goal(ax)
    _format_axis(ax)
    ax.set_xlabel(r"$x$ (km)")
    ax.set_ylabel(r"$y$ (km)")
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, label="surface elevation $h$", shrink=0.82, pad=0.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "fig_realistic")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    results = {}

    print("Experiment 1: analytic correctness ...")
    exp_correctness(results)

    print("Experiment 2: four-mode benchmark ...")
    benchmark = exp_benchmark(results)

    print("Experiment 3: regime / ablation sweep ...")
    regime = exp_regime(results)

    print("Experiment 4: resolution independence ...")
    M_grid, costs = exp_convergence(results)

    print("Experiment 5: multi-seed robustness ...")
    exp_robustness(results)

    print("Experiment 6: runtime vs grid solver ...")
    exp_runtime(results)

    print("Experiment 7: irregular sea benchmark ...")
    realistic = exp_realistic(results)

    print("Experiment 8: reverse-scenario benchmark ...")
    exp_reverse(results)

    print("Rendering figures ...")
    sc_c = make_scenario("combined")
    fig_scenario(sc_c)
    fig_paths(benchmark, sc_c)
    fig_benchmark(benchmark)
    fig_regime(regime)
    fig_convergence(M_grid, costs)
    fig_realistic(make_spectral_scenario("combined"), realistic)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    lines = ["# USV Riemannian Geodesic Planning -- Simulation Results\n",
             f"Seed={SEED}, L={L}, N_field={N_FIELD}, N_dijkstra={N_D}\n",
             "\n## Correctness\n"]
    c = results["correctness"]
    lines.append(f"- flat: Dijkstra {c['flat']['dijkstra']:.4f}, direct {c['flat']['direct_geodesic']:.4f}, "
                 f"straight {c['flat']['straight_line']:.4f}, FMM {c['flat']['fmm']:.4f}, "
                 f"LBR {c['flat']['lbr']:.4f} (Euclid {c['euclidean_ground_truth']:.4f})")
    lines.append(f"- constant current v=(0.4,0): analytic {c['constant_current']['analytic_reference']:.4f}, "
                 f"Dijkstra {c['constant_current']['dijkstra']:.4f}, direct {c['constant_current']['direct_geodesic']:.4f}, "
                 f"FMM {c['constant_current']['fmm']:.4f}, LBR {c['constant_current']['lbr']:.4f}")
    lines.append("\n## Benchmark (primary scenario, start=(1,1) goal=(9,9))\n")
    lines.append("| mode | dijkstra | direct | straight | ground-len | lateral-dev | weak-drift max | "
                 "time-reduction |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in ["flat", "surface", "current", "combined"]:
        b = benchmark[m]
        lines.append(f"| {m} | {b['dijkstra_cost']:.4f} | {b['direct_geodesic_cost']:.4f} | "
                     f"{b['straight_line_cost']:.4f} | {b['path_length_ground']:.4f} | "
                     f"{b['lateral_deviation']:.4f} | {b['weak_drift_max']:.4f} | "
                     f"{100*b['time_reduction_vs_straight']:+.2f}% |")
    lines.append("\n## Convergence\n")
    lines.append("| M | cost |\n|---|---|")
    for M, cst in zip(M_grid, costs):
        lines.append(f"| {M} | {cst:.6f} |")

    lines.append("\n## Robustness (multi-seed)\n")
    rb = results["robustness"]
    lines.append(f"- 10 seeds: min={rb['min']:.6f}, max={rb['max']:.6f}, "
                 f"std={rb['std']:.2e}, spread={100*rb['spread_rel']:.4f}%")

    lines.append("\n## Runtime vs grid eikonal solver\n")
    lines.append("| solver | resolution | cost | time (s) |\n|---|---|---|---|")
    for M, d in results["runtime"]["direct"].items():
        lines.append(f"| direct | M={M} | {d['cost']:.4f} | {d['time_s']:.3f} |")
    for N_d, g in results["runtime"]["grid"].items():
        lines.append(f"| grid Dijkstra (8c) | N={N_d} | {g['cost']:.4f} | {g['time_s']:.3f} |")
    for key, g in results["runtime"]["grid_wider"].items():
        lines.append(f"| grid Dijkstra ({g['connectivity']}c) | N={g['N']} | "
                     f"{g['cost']:.4f} | {g['time_s']:.3f} |")
    for key, g in results["runtime"]["fmm"].items():
        lines.append(f"| fast-marching ({g['stencil']}-stencil) | N={g['N']} | "
                     f"{g['cost']:.4f} | {g['time_s']:.3f} |")
    for N_d, g in results["runtime"]["lbr"].items():
        lines.append(f"| lattice-basis-reduction FMM (fast-swept) | N={N_d} | "
                     f"{g['cost']:.4f} | {g['time_s']:.3f} |")

    lines.append("\n## Realistic (Pierson--Moskowitz) benchmark\n")
    lines.append("| mode | direct | straight | ground-len | lateral-dev | weak-drift max | time-reduction |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in ["flat", "surface", "current", "combined"]:
        b = results["realistic_benchmark"][m]
        lines.append(f"| {m} | {b['direct_geodesic_cost']:.4f} | {b['straight_line_cost']:.4f} | "
                     f"{b['path_length_ground']:.4f} | {b['lateral_deviation']:.4f} | "
                     f"{b['weak_drift_max']:.4f} | {100*b['time_reduction_vs_straight']:+.2f}% |")

    lines.append("\n## Reverse benchmark (start=(9,9) goal=(1,1))\n")
    lines.append("| mode | direct | straight | ground-len | lateral-dev | weak-drift max | time-reduction |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in ["flat", "surface", "current", "combined"]:
        b = results["reverse_benchmark"][m]
        lines.append(f"| {m} | {b['direct_geodesic_cost']:.4f} | {b['straight_line_cost']:.4f} | "
                     f"{b['path_length_ground']:.4f} | {b['lateral_deviation']:.4f} | "
                     f"{b['weak_drift_max']:.4f} | {100*b['time_reduction_vs_straight']:+.2f}% |")

    with open("results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\nDone. Figures in figures/, results in results.json / results.md")
    print("\n--- Benchmark summary ---")
    for m in ["flat", "surface", "current", "combined"]:
        b = benchmark[m]
        print(f"{m:9s} direct={b['direct_geodesic_cost']:.4f}  "
              f"weak_drift={b['weak_drift_max']:.4f}  "
              f"reduction={100*b['time_reduction_vs_straight']:+.2f}%")
    print(f"\nWeak-drift condition |v|_g < 1: "
          f"max over all modes = {max(benchmark[m]['weak_drift_max'] for m in benchmark):.4f}")

    rb = results["robustness"]
    print(f"\nRobustness: 10-seed spread = {100*rb['spread_rel']:.4f}% "
          f"(min={rb['min']:.6f}, max={rb['max']:.6f})")
    rw = max(results["realistic_benchmark"][m]["weak_drift_max"]
             for m in results["realistic_benchmark"])
    print(f"Realistic scenario max |v|_g = {rw:.4f} (< 1 required)")
    for m in ["surface", "current", "combined"]:
        b = results["realistic_benchmark"][m]
        print(f"  realistic {m:8s} direct={b['direct_geodesic_cost']:.4f} "
              f"reduction={100*b['time_reduction_vs_straight']:+.2f}%")


if __name__ == "__main__":
    main()
