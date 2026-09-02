"""Sanity checks: solver correctness on cases with known analytic answers."""
import numpy as np

from metrics import Scenario
from solvers import (
    grid_dijkstra,
    direct_geodesic,
    straight_line_cost,
    polyline_cost,
)

START = np.array([1.0, 1.0])
GOAL = np.array([9.0, 9.0])
EUCLID = np.hypot(8.0, 8.0)  # 11.3137...


def report(name, cost, ref, tol):
    ok = abs(cost - ref) / ref < tol
    print(f"{name:28s} cost={cost:9.4f}  ref={ref:9.4f}  {'OK' if ok else 'FAIL'}")
    return ok


print("=== Test 1: flat, no wave/current -> Euclidean straight line ===")
sc = Scenario(L=10.0, N=129, mode="flat")
c_dij, path_dij, *_ = grid_dijkstra(sc, START, GOAL, N_d=129)
c_dir, path_dir = direct_geodesic(START, GOAL, sc, M=24, n_restarts=3, seed=1)
c_str = straight_line_cost(START, GOAL, sc)
report("dijkstra", c_dij, EUCLID, 0.03)
report("direct geodesic", c_dir, EUCLID, 0.01)
report("straight line", c_str, EUCLID, 0.001)

print("\n=== Test 2: constant current (Zermelo, straight-line geodesic) ===")
# constant current v=(0.4, 0.0); geodesic is a straight line in ground frame.
# cost along direction d=(8,8) with v=(0.4,0): use randers_flat formula.
from metrics import randers_flat_cost_vec
vx, vy = 0.4, 0.0
dvec = GOAL - START
ref2 = float(randers_flat_cost_vec(np.array([dvec]), np.array([vx]), np.array([vy]))[0])
# Build a scenario with a *constant* current by zeroing U (eddy) and U0=0.4.
sc2 = Scenario(L=10.0, N=129, mode="current", U=0.0, U0=0.4)
c2_dij, *_ = grid_dijkstra(sc2, START, GOAL, N_d=129)
c2_dir, p2 = direct_geodesic(START, GOAL, sc2, M=24, n_restarts=3, seed=1)
c2_str = straight_line_cost(START, GOAL, sc2)
report("dijkstra (const current)", c2_dij, ref2, 0.04)
report("direct geodesic (const current)", c2_dir, ref2, 0.01)
report("straight line (const current)", c2_str, ref2, 0.001)

print("\n=== Test 3: surface only -> geodesic >= Euclidean ===")
sc3 = Scenario(L=10.0, N=129, mode="surface")
c3_dir, p3 = direct_geodesic(START, GOAL, sc3, M=32, n_restarts=3, seed=1)
c3_str = straight_line_cost(START, GOAL, sc3)
print(f"surface geodesic = {c3_dir:.4f}, straight-line (on surface) = {c3_str:.4f}, "
      f"euclidean = {EUCLID:.4f}")
assert c3_dir >= EUCLID - 1e-6, "surface path must be >= euclidean"
assert c3_dir <= c3_str + 1e-6, "geodesic must be <= straight line"
print("OK: euclidean <= geodesic <= straight-line")

print("\nAll sanity checks completed.")
