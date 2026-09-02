"""Run the time-optimal planner on a *real* ocean-current snapshot.

This script extracts a regional surface-current field from a global, publicly
available netCDF product, maps it onto the paper's ``L x L`` (km) planning
domain, and runs the direct geodesic solver on it.  It is a self-contained
appendix-A4 artefact: it does not modify the core ``Scenario``/solver code, and
it writes a small ``.npz`` so the field can be reused without the 91 MB source.

Provenance (do not re-fabricate; cite exactly):
  Dataset   : Daily Global Sea Surface Ageostrophic Current (AC-PIDNN), 0.25 deg.
              Cui, G. & Cai, Z. (University of Macau).  DOI 10.5281/zenodo.19966716
  GitHub    : https://github.com/cgxcgxcgxcgx/AC-PIDNN  (documents the dataset)
  Download  : https://s3.jn1.is.shanhe.com/umdr2026/YYYY/MM/YYYY_MM_DD.nc
  Local file: data/2023_01_08.nc  (global 1440x720 grid, u/v in m/s)

Physical mapping: the Randers/Zermelo metric uses a *dimensionless* drift
``v`` with weak-drift condition ``|v| < 1`` (current speed as a fraction of the
USV's speed).  The real field (m/s) is divided by a reference USV speed
``U_ship``.  The Gulf Stream snapshot peaks near 1.6 m/s, so ``U_ship = 4 m/s``
gives ``max |v| ~ 0.40`` -- matching the synthetic ``current`` scenario's peak
``|v|_g = 0.39`` and validating that choice against real data.
"""

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

from metrics import randers_flat_cost_vec
from solvers import direct_geodesic, straight_line_cost

# Region of interest (Gulf Stream western-boundary current).
LON0, LON1 = -80.0, -60.0
LAT0, LAT1 = 30.0, 42.0
U_SHIP = 4.0          # reference USV speed [m/s] for the dimensionless drift
L = 10.0              # paper domain [0, L] x [0, L] (km)
N_FIELD = 257         # resample resolution (matches Scenario default)
START = np.array([1.0, 1.0])
GOAL = np.array([9.0, 9.0])

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NC_FILE = os.path.join(DATA_DIR, "2023_01_08.nc")
NPZ_FILE = os.path.join(DATA_DIR, "real_current_gulfstream.npz")


def load_gulfstream(nc_path=NC_FILE):
    """Read ``utotal``/``vtotal`` (m/s) over the Gulf Stream box, NaN -> 0.

    Returns ``(lon, lat, u, v)`` where ``u``, ``v`` are ``(nlon, nlat)``.
    Land/masked cells (``NaN``) are set to zero current; their fraction is
    reported so the coastal fill is not mistaken for a data value.
    """
    import netCDF4 as nc

    ds = nc.Dataset(nc_path)
    lon = ds.variables["lon"][:]
    lat = ds.variables["lat"][:]
    u = ds.variables["utotal"][:]
    v = ds.variables["vtotal"][:]
    ds.close()

    mlon = (lon >= LON0) & (lon <= LON1)
    mlat = (lat >= LAT0) & (lat <= LAT1)
    u = u[np.ix_(mlon, mlat)].astype(float)
    v = v[np.ix_(mlon, mlat)].astype(float)
    lon = lon[mlon]
    lat = lat[mlat]

    nan_frac = float(np.isnan(u).mean() + np.isnan(v).mean()) / 2.0
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    return lon, lat, u, v, nan_frac


def resample_to_domain(lon, lat, u, v, L=L, N=N_FIELD, U_ship=U_SHIP):
    """Interpolate the real field onto the ``N x N`` domain and non-dimensionalise.

    Returns ``(vx, vy)`` dimensionless ``(N, N)`` drift fields.
    """
    xs = np.linspace(0.0, L, N)
    ys = np.linspace(0.0, L, N)
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")

    # Physical (x, y) -> (lon, lat) affine map over the box.
    lon_q = LON0 + (LON1 - LON0) * Xg / L
    lat_q = LAT0 + (LAT1 - LAT0) * Yg / L

    interp_u = RegularGridInterpolator((lon, lat), u, bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator((lon, lat), v, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([lon_q.ravel(), lat_q.ravel()])
    vx = interp_u(pts).reshape(N, N) / U_ship
    vy = interp_v(pts).reshape(N, N) / U_ship
    return vx, vy


class RealCurrentScenario:
    """Minimal ``Scenario``-like wrapper over a real dimensionless drift field.

    Exposes the ``.L``, ``.cost_batch`` and ``.fields`` interface the solvers
    expect (flat base metric + real current), so ``direct_geodesic`` and
    ``grid_dijkstra`` run unchanged.
    """

    def __init__(self, vx, vy, L=L):
        self.L = L
        self.N = vx.shape[0]
        self._vx = vx
        self._vy = vy
        self._dx = L / (self.N - 1)

    def fields(self, pts):
        pts = np.asarray(pts, dtype=float)
        g = pts / self._dx
        i = g[:, 0]
        j = g[:, 1]
        vx = map_coordinates(self._vx, [i, j], order=1, mode="nearest")
        vy = map_coordinates(self._vy, [i, j], order=1, mode="nearest")
        return np.zeros_like(vx), np.zeros_like(vy), vx, vy

    def cost_batch(self, pts, xi):
        _, _, vx, vy = self.fields(pts)
        return randers_flat_cost_vec(xi, vx, vy)


def main():
    if not os.path.exists(NC_FILE):
        raise SystemExit(
            "Missing %s\nDownload it with:\n"
            "  curl -L -o %s https://s3.jn1.is.shanhe.com/umdr2026/2023/01/2023_01_08.nc"
            % (NC_FILE, NC_FILE)
        )

    lon, lat, u, v, nan_frac = load_gulfstream()
    spd = np.sqrt(u * u + v * v)

    print("Real ocean-current snapshot: AC-PIDNN 2023-01-08 (Gulf Stream box)")
    print("  box: lon [%.1f, %.1f], lat [%.1f, %.1f]  (%.2f deg grid)"
          % (LON0, LON1, LAT0, LAT1, lon[1] - lon[0]))
    print("  masked (land) cells: %.1f%%" % (100.0 * nan_frac))
    print("  surface current speed: min=%.3f  mean=%.3f  max=%.3f m/s"
          % (spd.min(), spd.mean(), spd.max()))
    print("  mean (u, v) = (%.3f, %.3f) m/s" % (u.mean(), v.mean()))

    vx, vy = resample_to_domain(lon, lat, u, v)
    peak = float(np.sqrt(vx * vx + vy * vy).max())
    print("  dimensionless drift |v| = |v_real| / U_ship (U_ship=%.1f m/s):"
          % U_SHIP)
    print("  peak |v| = %.3f  (weak-drift |v| < 1 holds)" % peak)

    np.savez(
        NPZ_FILE,
        vx=vx, vy=vy, L=L, N=N_FIELD, U_ship=U_SHIP,
        lon=lon, lat=lat, u_ms=u, v_ms=v,
        source="AC-PIDNN DOI 10.5281/zenodo.19966716",
    )
    print("  saved reusable field -> %s" % NPZ_FILE)

    sc = RealCurrentScenario(vx, vy)
    c_flat = straight_line_cost(START, GOAL, sc)
    c_dir, _ = direct_geodesic(START, GOAL, sc, M=32, n_restarts=6, seed=0)
    c_straight_flat = float(np.hypot(*(GOAL - START)))
    print("\nPlanner on real current (start=(1,1), goal=(9,9)):")
    print("  flat baseline (no current) = %.4f" % c_straight_flat)
    print("  straight line (real curr)  = %.4f" % c_flat)
    print("  direct geodesic (real cur) = %.4f" % c_dir)
    print("  time reduction = %+.2f%%" % (100.0 * (1.0 - c_dir / c_straight_flat)))


if __name__ == "__main__":
    main()
