"""Selling's algorithm for decomposing a 2x2 symmetric positive definite matrix.

This is a self-contained NumPy port of the 2D case of Mirebeau's
``Selling.Decomposition`` (SIAM J. Numer. Anal. 52(4), 2014).  A symmetric
positive definite matrix ``m`` is written as

    m = sum_k coefs[k] * outer(offsets[k], offsets[k]),

with ``coefs[k] >= 0`` and ``offsets[k]`` an integer 2-vector.  The offsets are
the perpendiculars of an *m-obtuse superbase*: three integer vectors that sum to
zero and meet pairwise at obtuse (non-positive) angles in the ``m`` inner
product.  They form the metric-adapted stencil of a lattice-basis-reduction
(LBR) fast-marching scheme, replacing a fixed axis-aligned stencil so that the
discretisation carries no directional (metrication) bias.

Only the scalar (single-matrix) 2D case is needed here; ``grid_decomposition``
loops it over an ``(N, N)`` field of matrices for the FMM.
"""

import numpy as np


def _dot_vav(u, m, w):
    """Return ``u.T @ m @ w`` for 2-vectors ``u, w`` and 2x2 ``m``."""
    return float(u[0] * (m[0, 0] * w[0] + m[0, 1] * w[1])
                 + u[1] * (m[1, 0] * w[0] + m[1, 1] * w[1]))


def _obtuse_superbase_2d(m):
    """Reduce the canonical superbase to an ``m``-obtuse superbase.

    The canonical superbase columns are ``e0 = (-1,-1), e1 = (1,0), e2 = (0,1)``
    (they sum to zero and are pairwise obtuse in the Euclidean inner product).
    The Selling reduction loop tilts the basis until it is obtuse in the ``m``
    inner product, which is the condition for all decomposition coefficients to
    be non-negative.
    """
    sb = np.array([[-1.0, 1.0, 0.0],
                   [-1.0, 0.0, 1.0]], dtype=float)
    cycle = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    iter_reduced = 0
    for it in range(100):
        i, j, k = cycle[it % 3]
        acute = _dot_vav(sb[:, i], m, sb[:, j]) > 0.0
        if acute:
            sb[:, k] = sb[:, i] - sb[:, j]
            sb[:, i] = -sb[:, i]
            iter_reduced = 0
        elif iter_reduced < 3:
            iter_reduced += 1
        else:
            return sb
    raise ValueError("Selling reduction did not terminate in 100 iterations")


def selling_decomposition_2d(m):
    """Decompose SPD 2x2 ``m`` into ``(coefs, offsets)``.

    Returns ``coefs`` (shape ``(3,)``, non-negative) and ``offsets`` (shape
    ``(2, 3)``, integer columns) such that ``m = sum_k coefs[k] *
    outer(offsets[:, k], offsets[:, k])``.
    """
    m = np.asarray(m, dtype=float)
    sb = _obtuse_superbase_2d(m)

    coefs = np.zeros(3)
    for idx, (i, j, k) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
        # coef[i] = -<e_j, m e_k>  (cyclic), cf. _Decomposition2 in agd.
        coefs[i] = -_dot_vav(sb[:, j], m, sb[:, k])

    # offsets[:, i] = perp(e_i) = (-e_i[1], e_i[0]).
    offsets = np.array([[-sb[1, i], sb[0, i]] for i in range(3)]).T.astype(int)
    return coefs, offsets


def grid_decomposition(Gxx, Gxy, Gyy):
    """Selling decomposition of a field of cotangent metrics.

    ``Gxx, Gxy, Gyy`` are ``(N, N)`` arrays of the inverse surface metric.
    Returns ``(offs, coef)`` where ``offs[i, j]`` is the list of three integer
    offset tuples and ``coef[i, j]`` the three non-negative coefficients.
    """
    N = Gxx.shape[0]
    offs = np.empty((N, N), dtype=object)
    coef = np.empty((N, N), dtype=object)
    for i in range(N):
        for j in range(N):
            m = np.array([[Gxx[i, j], Gxy[i, j]],
                          [Gxy[i, j], Gyy[i, j]]])
            c, o = selling_decomposition_2d(m)
            coef[i, j] = c
            offs[i, j] = [(int(o[0, k]), int(o[1, k])) for k in range(3)]
    return offs, coef
