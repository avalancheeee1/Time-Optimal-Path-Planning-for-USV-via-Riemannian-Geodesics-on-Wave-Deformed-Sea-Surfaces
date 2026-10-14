"""Perturbation-theory check: the wave surface enters the time-optimal cost at
O(eps^2), with a closed-form second-order coefficient.

The unified Randers metric has base metric  g_ij = delta_ij + (d_i h)(d_j h).
Writing the elevation as  h = eps * eta  (eps = wave-slope scale), the base
metric is  g = delta + eps^2 (grad eta)(grad eta)^T, an EVEN function of eps.
The Randers cost  F = sqrt(a_ij xi^i xi^j) + b_i xi^i  is therefore even in eps,
so the optimal travel time  T*(eps) = min_gamma int F  is even too:

    T*(eps) = T0 + eps^2 T2 + O(eps^4),     T1 = 0 identically.

Because d gamma_eps / d eps |_{eps=0} = 0 (the geodesic itself does not move to
first order under an even perturbation), the second-order coefficient is the
integral of the metric's second variation along the UNPERTURBED (current-only)
geodesic gamma_0 -- no re-optimization is needed:

    T2 = int_{gamma_0} F2 dt.

A direct expansion of (3) in eps^2 gives  F2 = alpha2^2/(2 alpha0) + beta2, with

    alpha0^2 = |xi|^2/lambda0 + e^2/lambda0^2,
    alpha2^2 = d^2/lambda0 + c^2 |xi|^2/lambda0^2
               + 2 c d e/lambda0^2 + 2 c^2 e^2/lambda0^3,
    beta2    = - c d/lambda0 - c^2 e/lambda0^2,

    c = <grad eta, v>  (wave-current alignment),
    d = <grad eta, xi> (path motion against the wave gradient),
    e = <v, xi>,
    lambda0 = 1 - |v|^2.

F2 splits into a SURFACE-ONLY part (the d^2 term, which survives v -> 0 and is
the known first-order surface lengthening) and a COUPLING part (every term with
c, which vanishes when v = 0).  This is the "non-additive" wave-current coupling
of the benchmark, now as an exact second-order formula.

This script runs three checks on the deterministic seed-0 scenario:
  1. T[gamma_0, eps] (path held fixed) is quadratic in eps with no linear term,
     and its coefficient equals the closed-form int F2.
  2. The re-optimized T*(eps) (full solver, surface amplitude scaled by eps)
     matches T0 + eps^2 T2 to O(eps^4), confirming the stationarity argument.
  3. The surface-only vs. coupling split, and the dimensionless criterion
     eps^2 |T2| / (first-order current saving) that decides when the surface
     must be modeled.
"""
import numpy as np

from metrics import (Scenario, current_field, surface_grad,
                     randers_surface_cost)
from solvers import direct_geodesic, straight_line_cost
import experiments as E


def _fields_at(x, y):
    """Analytic (vx, vy, hx, hy) at a point, matching Scenario's defaults."""
    vx, vy = current_field(np.array([x]), np.array([y]))
    hx, hy = surface_grad(np.array([x]), np.array([y]))
    return float(vx[0]), float(vy[0]), float(hx[0]), float(hy[0])


def path_cost_at_scale(path, eps):
    """Exact Randers cost of `path` with g = delta + (eps grad eta)^2 (current
    kept, no re-optimization)."""
    total = 0.0
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        xi = b - a
        xm, ym = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        vx, vy, hx, hy = _fields_at(xm, ym)
        gxx = 1.0 + (eps * hx) ** 2
        gxy = (eps * hx) * (eps * hy)
        gyy = 1.0 + (eps * hy) ** 2
        total += randers_surface_cost(xi, (vx, vy), gxx, gxy, gyy)
    return total


def T2_closed_form(path):
    """int_{gamma_0} F2 dt, split into (surface-only, coupling)."""
    surf = 0.0
    coup = 0.0
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        xi = b - a
        xm, ym = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        vx, vy, hx, hy = _fields_at(xm, ym)
        c = hx * vx + hy * vy
        d = hx * xi[0] + hy * xi[1]
        e = vx * xi[0] + vy * xi[1]
        xi2 = xi[0] ** 2 + xi[1] ** 2
        lam0 = 1.0 - (vx * vx + vy * vy)
        alpha0 = np.sqrt(xi2 / lam0 + e * e / (lam0 * lam0))
        # surface-only: the d^2 term (survives v -> 0), the known arc-length
        # lengthening.  Kept out of the coupling part to avoid double-counting.
        surf += d * d / (2.0 * alpha0 * lam0)
        # coupling: the c-dependent terms in alpha2^2 (excluding d^2) + beta2.
        a2_coup = (c * c * xi2 / lam0 ** 2
                   + 2.0 * c * d * e / lam0 ** 2
                   + 2.0 * c * c * e * e / lam0 ** 3)
        b2 = -c * d / lam0 - c * c * e / lam0 ** 2
        coup += a2_coup / (2.0 * alpha0) + b2
    return surf, coup


def _scaled_scenario(eps):
    """Combined scenario with surface amplitude scaled by eps (surface_height
    and surface_grad are linear in A, so A -> eps * A scales the gradient)."""
    return Scenario(L=E.L, N=E.N_FIELD, mode="combined",
                    A=(0.5 * eps, 0.3 * eps))


def main():
    sc0 = Scenario(L=E.L, N=E.N_FIELD, mode="current")
    T0, gamma0 = direct_geodesic(E.START, E.GOAL, sc0, M=64,
                                 n_restarts=6, seed=E.SEED)
    c_str = straight_line_cost(E.START, E.GOAL, sc0)
    first_order = c_str - T0              # current-only saving (first-order)
    print(f"gamma_0 (current-only): T0 = {T0:.6f},  straight = {c_str:.6f}")
    print(f"first-order current effect (saving) = {first_order:+.6f}")

    T2_surf, T2_coup = T2_closed_form(gamma0)
    T2 = T2_surf + T2_coup
    print(f"\nclosed-form T2: surf = {T2_surf:+.6f}   "
          f"coupl = {T2_coup:+.6f}   total = {T2:+.6f}")

    # --- check 1: path-fixed quadratic scaling, no linear term ---------------
    eps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    cost = np.array([path_cost_at_scale(gamma0, e) for e in eps])
    d = cost - cost[0]
    print("\n[check 1] T[gamma_0, eps] - T[gamma_0, 0]  (path fixed)")
    print("   eps    dT            dT/eps^2")
    for e, dd in zip(eps, d):
        if e > 0:
            print(f"   {e:4.2f}  {dd:+.8f}   {dd / e**2:+.6f}")
        else:
            print(f"   {e:4.2f}  {dd:+.8f}")
    coef_fd = d[1] / eps[1] ** 2
    print(f"   finite-diff coeff = {coef_fd:+.6f}   vs  closed-form T2 = {T2:+.6f}"
          f"   (rel diff {abs(T2 - coef_fd) / abs(T2):.2e})")

    # --- check 2: re-optimized T*(eps) ---------------------------------------
    print("\n[check 2] re-optimized T*(eps) vs T0 + eps^2 * T2")
    for e in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sc = _scaled_scenario(e)
        Te, _ = direct_geodesic(E.START, E.GOAL, sc, M=64,
                                n_restarts=6, seed=E.SEED)
        pred = T0 + e * e * T2
        print(f"   eps={e:4.2f}  T*={Te:.6f}  pred={pred:.6f}  "
              f"resid={Te - pred:+.2e}")

    # --- check 3: dimensionless criterion ------------------------------------
    print("\n[check 3] surface-only vs coupling, and the decision criterion")
    print(f"   surface-only T2 / |coupling T2| = {abs(T2_surf)/abs(T2_coup):.3f}")
    print(f"   eps^2 |T2| vs first-order current effect:")
    for e in [0.3, 0.5, 0.8, 1.0]:
        ratio = e * e * abs(T2) / abs(first_order)
        print(f"     eps={e:4.2f}  eps^2|T2|/|saving| = {ratio:.4f}  "
              f"({'surface matters' if ratio > 0.1 else 'surface negligible'})")


if __name__ == "__main__":
    main()
