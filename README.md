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

Beyond the core benchmark, four supplementary experiments (P1--P4) carry the
headline contributions: a heading-dependent Finsler wave metric (P1), an
independent geodesic-ODE optimality cross-check (P2), robustness across current
geometry and start/goal pair (P3), and a real wave-plus-current dual-field
validation (P4). Each writes its own `results_*.json`.

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
├── experiment_wave.py             # P1: heading-dependent Finsler wave metric
├── experiment_geodesic_ode.py     # P2: geodesic ODE two-point shooting cross-check
├── experiment_robustness.py       # P3: robustness across geometry and start/goal pair
├── experiment_real_wave_current.py  # P4: real wave + current dual-field validation
├── test_fmm.py            # unit tests for the FMM solver and geodesic correctness
├── test_sanity.py         # sanity tests for the metric / scenario
├── verify_reproducibility.py  # re-runs experiments and checks byte-for-byte reproducibility
├── requirements.txt       # pinned environment (Python 3.13.7)
├── results.json           # machine-readable results for the core benchmark
├── results_wave.json              # P1 machine-readable results
├── results_geodesic_ode.json      # P2 machine-readable results
├── results_robustness.json        # P3 machine-readable results
├── results_real_wave_current.json # P4 machine-readable results
├── results.md             # human-readable summary of the same numbers
└── data/
    ├── real_current_gulfstream.npz   # extracted Gulf Stream snapshot (shipped for convenience)
    └── ww3_gulfstream_20230108.csv   # WW3 wave-model fields (P4, shipped)
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

## Additional experiments (P1--P4)

Each experiment is a standalone script that writes its own `results_*.json` with
a deterministic seed; run them from `simulation/`:

```bash
python experiment_wave.py             # P1 -> results_wave.json
python experiment_geodesic_ode.py     # P2 -> results_geodesic_ode.json
python experiment_robustness.py       # P3 -> results_robustness.json
python experiment_real_wave_current.py  # P4 -> results_real_wave_current.json
```

- **P1 — heading-dependent wave metric** (`experiment_wave.py`). Replaces the
  symmetric surface arc length `g_ij = δ_ij + (∂_i h)(∂_j h)` with the
  *irreversible* Finsler metric `F(x,ξ) = |ξ|/s(x,θ(ξ))`,
  `s(x,θ) = 1 − ε(x)(1 − cos(θ−ψ(x)))/2`, where `ψ(x)` is the local wave
  propagation direction and `ε(x)` the speed-reduction amplitude. Head seas slow
  the vessel and following seas speed it, isolating the directional resistance
  from pure geometry.
- **P2 — geodesic-ODE cross-check** (`experiment_geodesic_ode.py`). Independently
  characterises the same time-optimal path via Pontryagin's minimum principle: a
  first-order ODE in position and costate, solved as a two-point boundary-value
  problem by shooting on the initial costate. Agreement with the direct solver
  rules out a spurious local minimum or a discretisation artefact.
- **P3 — robustness** (`experiment_robustness.py`). Re-runs the four-mode
  benchmark on a *different* current geometry (a meandering Gulf-Stream-like
  eastward jet) and a *different* start/goal pair that crosses the jet, verifying
  the same qualitative behaviour holds away from the primary scenario.
- **P4 — real wave + current dual field** (`experiment_real_wave_current.py`).
  Pairs the real AC-PIDNN current (Gulf Stream, 2023-01-08) with a real WW3
  wave field from the same region and date, validating the heading-dependent
  metric of P1 against real data (see the [Data](#data) section).

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

The real wave field used in the dual-field validation (P4) is the **WW3** global
wave model, served by PacIOOS (U. Hawaii) ERDDAP. The single snapshot used in the
paper — the Gulf Stream box lon 280–300° E, lat 30–42° N on 2023-01-08T00:00Z,
at 0.5° resolution (significant wave height `Thgt`, mean direction `Tdir`, peak
period `Tper`) — is shipped as `data/ww3_gulfstream_20230108.csv` (1025 data rows).
At this resolution the wave field is spatially uniform over the 10 km planning
domain, so the wave direction and severity enter the metric as single constants.

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
data used in the real-data figure is CC-BY-4.0, and the WW3 wave-model fields are
served openly by PacIOOS (U. Hawaii) ERDDAP (see the [Data](#data) section).
