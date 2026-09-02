"""Verification gates for the Randers fast-marching solver.

Run from the simulation directory:

    python test_fmm.py

Each gate now *asserts* a deterministic value instead of only printing it, so a
regression exits non-zero.  The tolerances are tight enough to catch the one bug
that matters here --- the asymmetric edge-move sign error, which shifts the
constant-current cost from 9.1329 to 9.2258 and the combined cost from 9.9669 to
9.9891 --- and loose enough to ignore floating-point noise.

Gate 1 (flat): start/goal are snapped to the grid, so the optimal path is the
45-degree diagonal.  The 8- and 24-neighbor stencils price that diagonal edge
exactly and must recover the snapped Euclidean distance 11.3432.

Gate 2 (constant current v=(0.4,0)): the geodesic is again the straight diagonal,
so the wider stencils must recover the snapped cost ~9.1329 (the analytic
reference 9.1092 is for the *unsnapped* endpoints).

Gate 3 (combined, varying surface + eddy current): the geodesic is curved, so
8-connected Dijkstra accumulates metrication error (10.5392); the upwind FMM
should approach the continuous optimum 9.8750 (direct solver).

Gate 4: the dual-norm identity F*(grad F(xi)) = 1, checked numerically for
random surface metrics and currents (ties the signed update back to the metric).

Gate 5 (convergence): pins the FMM values reported in the paper's runtime table
(Table VI).  The N=97 24-neighbor value is intentionally not asserted: it
undershoots the continuous optimum only because the endpoints snap onto a coarser
grid, an artifact documented in the paper.
"""

import numpy as np

from experiments import make_scenario, START, GOAL, N_D
from solvers import grid_dijkstra, direct_geodesic
from metrics import randers_flat_cost_vec, randers_surface_cost, surface_metric
from fmm import fmm_randers, fmm_randers_lbr

TOL = 5e-4  # deterministic costs reproduce to ~1e-12; this is generous.


def _check(name, got, want, tol=TOL):
    """Assert ``got`` is within ``tol`` of ``want``, printing a PASS/FAIL line."""
    ok = abs(got - want) <= tol
    print("%-5s %-30s got=%.6f  want=%.6f" % ("PASS" if ok else "FAIL", name, got, want))
    assert ok, "%s: got %.6f, want %.6f (tol=%.1e)" % (name, got, want, tol)


def gate_flat():
    sc = make_scenario("flat")
    c_dij, *_ = grid_dijkstra(sc, START, GOAL, N_d=N_D)
    c_dir, _ = direct_geodesic(START, GOAL, sc, M=32, n_restarts=6, seed=0)
    _check("flat dij8", c_dij, 11.3432)
    _check("flat direct", c_dir, 11.3137)
    for st in (4, 8, 24):
        c, *_ = fmm_randers(sc, START, GOAL, N_d=N_D, stencil=st)
        if st in (8, 24):
            _check("flat fmm(%d)" % st, c, 11.3432)
        else:
            print("info  flat fmm(4) = %.4f  (O(h) diagonal bias, not asserted)" % c)


def gate_constant():
    sc = make_scenario("current", U=0.0, U0=0.40)
    c_dij, *_ = grid_dijkstra(sc, START, GOAL, N_d=N_D)
    dvec = (GOAL - START)[None, :]
    ref = float(randers_flat_cost_vec(dvec, np.array([0.40]), np.array([0.0]))[0])
    _check("const dij8", c_dij, 9.1329)
    _check("const analytic (unsnapped)", ref, 9.1092)
    for st in (4, 8, 24):
        c, *_ = fmm_randers(sc, START, GOAL, N_d=N_D, stencil=st)
        if st in (8, 24):
            _check("const fmm(%d)" % st, c, 9.1329)
        else:
            print("info  const fmm(4) = %.4f  (O(h) diagonal bias, not asserted)" % c)


def gate_combined():
    sc = make_scenario("combined")
    c_dir, _ = direct_geodesic(START, GOAL, sc, M=32, n_restarts=6, seed=0)
    c_dij8, *_ = grid_dijkstra(sc, START, GOAL, N_d=N_D)
    c_dij24, *_ = grid_dijkstra(sc, START, GOAL, N_d=N_D, connectivity=24)
    _check("combined direct", c_dir, 9.8750)
    _check("combined dij8", c_dij8, 10.5392)
    _check("combined dij24", c_dij24, 9.9242)
    for st in (4, 8, 24):
        c, *_ = fmm_randers(sc, START, GOAL, N_d=N_D, stencil=st)
        if st == 4:
            print("info  combined fmm(4) = %.4f  (O(h) diagonal bias, not asserted)" % c)
        elif st == 8:
            _check("combined fmm8", c, 9.9669)
        else:
            _check("combined fmm24", c, 9.9205)


def gate_dual():
    """F*(grad F(xi)) = 1 for the Randers cost (Euler homogeneity)."""
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        hx = rng.uniform(-0.3, 0.3)
        hy = rng.uniform(-0.3, 0.3)
        vx = rng.uniform(-0.4, 0.4)
        vy = rng.uniform(-0.4, 0.4)
        gxx, gxy, gyy, Gxx, Gxy, Gyy, _ = surface_metric(np.array([hx]), np.array([hy]))
        xi = np.array([rng.uniform(0.2, 1.5), rng.uniform(0.2, 1.5)])

        def F(x):
            return randers_surface_cost(x, (vx, vy), float(gxx[0]), float(gxy[0]),
                                        float(gyy[0]))

        eps = 1e-6
        grad = np.array([(F(xi + eps * np.array([1, 0])) - F(xi - eps * np.array([1, 0]))) / (2 * eps),
                         (F(xi + eps * np.array([0, 1])) - F(xi - eps * np.array([0, 1]))) / (2 * eps)])
        px, py = grad
        fstar = px * vx + py * vy + np.sqrt(Gxx[0] * px * px
                                            + 2 * Gxy[0] * px * py + Gyy[0] * py * py)
        max_err = max(max_err, abs(fstar - 1.0))
    print("dual max |F*(grad F)-1| = %.2e  (over 200 random metrics)" % max_err)
    assert max_err < 1e-8, "dual-norm identity violated: max_err = %.2e" % max_err


def gate_convergence():
    """Pin the FMM values reported in the paper's runtime table (Table VI)."""
    sc = make_scenario("combined")
    c_dir, _ = direct_geodesic(START, GOAL, sc, M=32, n_restarts=6, seed=0)
    _check("convergence direct", c_dir, 9.8750)

    # (N, stencil) -> cost, exactly as reported in Table VI.  The N=97 24-neighbor
    # entry is deliberately absent (endpoint-rounding undershoot, see docstring).
    reported = {
        (97, 8): 9.8816,
        (193, 8): 9.9669,
        (385, 8): 9.9390,
        (193, 24): 9.9205,
        (385, 24): 9.9139,
    }
    for N in (97, 193, 385):
        row = "N=%d:" % N
        for st in (8, 24):
            c, *_ = fmm_randers(sc, START, GOAL, N_d=N, stencil=st)
            row += "  fmm%d=%.4f" % (st, c)
            if (N, st) in reported:
                _check("convergence fmm%d N=%d" % (st, N), c, reported[(N, st)])
        c8, *_ = grid_dijkstra(sc, START, GOAL, N_d=N)
        c24, *_ = grid_dijkstra(sc, START, GOAL, N_d=N, connectivity=24)
        row += "  dij8=%.4f  dij24=%.4f" % (c8, c24)
        print("        " + row)


def gate_lbr():
    """Lattice-basis-reduction (LBR) FMM, Mirebeau's Selling scheme.

    The Randers dual eikonal is cast to the shifted-Riemannian (Zermelo) form and
    the SPD tensor ``H = G - v v^T`` is Selling-decomposed into integer offsets.
    For the paper's *near-isotropic* scenarios this decomposition collapses to the
    two cardinal offsets (the anti-diagonal carries zero weight), so the LBR update
    reduces to the classic 4-neighbour Godunov scheme and carries the same O(h)
    diagonal bias.  The two assertions below pin that reduction:

    * flat LBR == fmm(4): the isotropic Selling stencil is exactly the cardinal
      pair, recovering the 4-neighbour update (11.4313, not the 8-neighbour 11.3432).
    * current LBR does not undershoot the analytic reference (the shifted metric is
      non-causal for a single march, so the monotone scheme is fast-swept to a fixed
      point; a single-pass march undershoots and is therefore incorrect).
    """
    sc_flat = make_scenario("flat")
    c4, *_ = fmm_randers(sc_flat, START, GOAL, N_d=N_D, stencil=4)
    c_lbr, *_ = fmm_randers_lbr(sc_flat, START, GOAL, N_d=N_D)
    _check("lbr flat == fmm(4)", c_lbr, c4, tol=5e-5)

    sc_cur = make_scenario("current", U=0.0, U0=0.40)
    ref = 9.1092  # analytic reference (unsnapped)
    c_lbr_c, *_ = fmm_randers_lbr(sc_cur, START, GOAL, N_d=N_D)
    _check("lbr current (monotone, >= analytic)", c_lbr_c, 9.232856, tol=5e-4)
    assert c_lbr_c >= ref, "LBR undershoots the analytic reference (non-monotone)"

    # Combined scenario: the LBR is a 4-neighbour scheme here, so it lands between
    # fmm(8) and fmm(4) and does not beat the wider fixed stencils.  Pinned at N=97
    # (the N=193 value 9.9941 is recorded in results.json).
    sc_cmb = make_scenario("combined")
    c_lbr_cmb, *_ = fmm_randers_lbr(sc_cmb, START, GOAL, N_d=97)
    _check("lbr combined N=97 (first-order)", c_lbr_cmb, 9.937686, tol=5e-4)


if __name__ == "__main__":
    gate_flat()
    gate_constant()
    gate_combined()
    gate_dual()
    gate_convergence()
    gate_lbr()
    print("\nAll FMM gates passed.")
