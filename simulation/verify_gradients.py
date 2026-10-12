"""Verify the analytic geodesic gradients against central differences.

For every scenario mode, builds a random polyline and checks
``polyline_gradient`` (gradients.py) against a central-difference gradient of
``polyline_cost`` (solvers.py).  A pass here is the safety gate before the
analytic Jacobian is used in any production run.
"""

import numpy as np

from metrics import Scenario
from solvers import polyline_cost
from gradients import polyline_gradient


L = 10.0
N = 65
START = np.array([1.0, 1.0])
GOAL = np.array([9.0, 9.0])
MODES = ["flat", "surface", "current", "combined", "wave", "wave_current"]


def random_polyline(M, rng):
    z = START + (GOAL - START) * np.linspace(0.0, 1.0, M + 2)[:, None]
    z[1:-1] += rng.normal(0.0, 0.3, (M, 2))
    z[1:-1] = np.clip(z[1:-1], 0.2, L - 0.2)
    return z


def central_diff_grad(points, scenario, h=1e-6):
    M = len(points) - 2
    grad = np.zeros((M, 2))
    for k in range(1, M + 1):
        for a in (0, 1):
            pp = points.copy(); pm = points.copy()
            pp[k, a] += h; pm[k, a] -= h
            grad[k - 1, a] = (polyline_cost(pp, scenario)
                              - polyline_cost(pm, scenario)) / (2 * h)
    return grad


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    for mode in MODES:
        sc = Scenario(L=L, N=N, mode=mode)
        for trial in range(4):
            M = int(rng.integers(4, 12))
            pts = random_polyline(M, rng)
            g_a = polyline_gradient(pts, sc)
            g_n = central_diff_grad(pts, sc)
            err = np.abs(g_a - g_n).max()
            rel = err / (np.abs(g_n).max() + 1e-12)
            worst = max(worst, err)
            print(f"{mode:>12s}  M={M:2d}  max|ana-num|={err:.3e}  rel={rel:.3e}")
            assert err < 1e-5, (
                f"FAIL {mode} M={M}: gradient mismatch {err:.3e}")
    print(f"\nAll {len(MODES)} modes passed.  Worst absolute error = {worst:.3e}")


if __name__ == "__main__":
    main()
