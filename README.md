# Time-Optimal Path Planning for USVs via Riemannian Geodesics on Wave-Deformed Sea Surfaces

Reproducible code and data for

> **X. Li, J. Zhang, P. Zhang, and L. Tan**, "Time-Optimal Path Planning for
> Unmanned Surface Vehicles via Riemannian Geodesics on Wave-Deformed Sea
> Surfaces," under review.

This repository reproduces every numeric result, table, and figure in the
paper. The planning problem is cast as a **Zermelo navigation problem** on a
wave-deformed sea surface: the medium drift is the surface current, the
configuration space is a Riemannian graph of the wavy free surface, and the
time-optimal path is the geodesic of the resulting **Randers metric**. Geodesics
are computed with a from-scratch fast-marching (FMM) solver, optionally with a
Selling lattice-basis-reduction preconditioner.

## Repository layout

```
simulation/
├── experiments.py         # main entry point; runs all experiments, writes results.json / results.md / figures/
├── metrics.py             # Riemannian surface metric, current field, Randers cost (combined mode)
├── solvers.py             # direct-geodesic & straight-line planners, graph Dijkstra
├── fmm.py                 # Randers fast-marching solver (8/24-stencil) + LBR variant
├── selling.py             # Selling's lattice-basis-reduction decomposition (2-D)
├── fig_reverse.py         # Figure: reversed-current / cross-current benchmark
├── real_current.py        # one-time extraction of the real ocean-current snapshot -> data/*.npz
├── fig_real_current.py    # Figure: time-optimal path over the real Gulf Stream field
├── test_fmm.py            # unit tests for the FMM solver and geodesic correctness
├── test_sanity.py         # sanity tests for the metric / scenario
├── verify_reproducibility.py  # re-runs experiments and checks byte-for-byte reproducibility
├── requirements.txt       # pinned environment (Python 3.13.7)
├── results.json           # machine-readable results for all tables/figures
├── results.md             # human-readable summary of the same numbers
└── data/
    └── real_current_gulfstream.npz   # extracted Gulf Stream snapshot (shipped for convenience)
```

## Reproduce

```bash
cd simulation
python -m pip install -r requirements.txt
python experiments.py        # runs all experiments; writes results.json, results.md, figures/
```

All experiments use a deterministic seed, so every scientific quantity in
`results.json` (costs, path reductions, convergence, multi-seed robustness)
regenerates **byte-for-byte**. Figures are written to `simulation/figures/`
(PDF/SVG/PNG).

Run the tests:

```bash
python -m pytest test_fmm.py test_sanity.py -q
```

## Reproducibility

```bash
python verify_reproducibility.py
```

re-runs `experiments.py` and asserts that `results.json` reproduces
byte-for-byte. The only fields excluded from this guarantee are the wall-clock
`time_s` measurements under `results.runtime` (machine/load-dependent), which are
stripped before hashing. The canonical SHA-256 of the deterministic projection
(`results.json` with every `time_s` key removed) is:

```
4ce121b810b080c2e0a9d09b77cd63537b7d32c2ee54440a61b78d7ff9b8246d
```

## Data

The real ocean-current field in the paper is the **AC-PIDNN** daily sea-surface
current (0.25°, global, u/v in m/s):

- **Dataset**: Cui, G. & Cai, Z. (University of Macau) — *Daily Global Sea
  Surface Ageostrophic Current (AC-PIDNN)*, DOI
  [`10.5281/zenodo.19966716`](https://doi.org/10.5281/zenodo.19966716)
  (CC-BY-4.0). Documented at <https://github.com/cgxcgxcgxcgx/AC-PIDNN>.
- The **91 MB** raw NetCDF snapshot (`2023_01_08.nc`) is **not** shipped. Fetch it
  and re-extract the small reusable field with:

  ```bash
  curl -L -o data/2023_01_08.nc https://s3.jn1.is.shanhe.com/umdr2026/2023/01/2023_01_08.nc
  python real_current.py        # writes data/real_current_gulfstream.npz
  ```

  (This step additionally requires `netCDF4`; the extracted `real_current_gulfstream.npz`
  shipped in this repo is sufficient to run `fig_real_current.py` and everything
  else without `netCDF4`.)

## Environment

| Component | Version |
|-----------|---------|
| Python    | 3.13.7  |
| NumPy     | 2.2.3   |
| SciPy     | 1.16.1  |
| Matplotlib| 3.10.8  |

Reproduced on an AMD Ryzen 7 9800X3D with 32 GB RAM under Windows 11.

## License

Code is released under the [MIT License](LICENSE). The AC-PIDNN ocean-current
data used in the real-data figure is CC-BY-4.0 (see the [Data](#data) section).
