# USV Riemannian Geodesic Planning -- Simulation Results

Seed=0, L=10.0, N_field=257, N_dijkstra=193


## Correctness

- flat: Dijkstra 11.3432, direct 11.3137, straight 11.3137, FMM 11.3432, LBR 11.4313 (Euclid 11.3137)
- constant current v=(0.4,0): analytic 9.1092, Dijkstra 9.1329, direct 9.1092, FMM 9.1329, LBR 9.2329

## Benchmark (primary scenario, start=(1,1) goal=(9,9))

| mode | dijkstra | direct | straight | ground-len | lateral-dev | weak-drift max | time-reduction |
|---|---|---|---|---|---|---|---|
| flat | 11.3432 | 11.3137 | 11.3137 | 11.3137 | 0.0000 | 0.0000 | -0.00% |
| surface | 11.6367 | 11.6050 | 11.6667 | 11.3176 | 0.0908 | 0.0000 | -2.57% |
| current | 10.3407 | 9.6864 | 10.4231 | 11.7509 | 1.4876 | 0.3926 | +14.38% |
| combined | 10.5392 | 9.8750 | 10.6975 | 11.7793 | 1.5434 | 0.3926 | +12.72% |

## Convergence

| M | cost |
|---|---|
| 4 | 9.874847 |
| 8 | 9.877783 |
| 16 | 9.875775 |
| 32 | 9.875033 |
| 64 | 9.874706 |

## Robustness (multi-seed)

- 10 seeds: min=9.875020, max=9.875093, std=2.45e-05, spread=0.0007%

## Runtime vs grid eikonal solver

| solver | resolution | cost | time (s) |
|---|---|---|---|
| direct | M=8 | 9.8778 | 0.571 |
| direct | M=16 | 9.8758 | 1.228 |
| direct | M=32 | 9.8750 | 1.757 |
| grid Dijkstra (8c) | N=97 | 10.4026 | 0.462 |
| grid Dijkstra (8c) | N=193 | 10.5392 | 1.870 |
| grid Dijkstra (8c) | N=385 | 10.5395 | 7.618 |
| grid Dijkstra (16c) | N=193 | 9.9245 | 3.707 |
| grid Dijkstra (16c) | N=385 | 9.9254 | 15.072 |
| grid Dijkstra (24c) | N=193 | 9.9242 | 5.590 |
| grid Dijkstra (24c) | N=385 | 9.9253 | 22.528 |
| fast-marching (8-stencil) | N=97 | 9.8816 | 0.254 |
| fast-marching (8-stencil) | N=193 | 9.9669 | 0.993 |
| fast-marching (8-stencil) | N=385 | 9.9390 | 4.095 |
| fast-marching (24-stencil) | N=97 | 9.7944 | 0.600 |
| fast-marching (24-stencil) | N=193 | 9.9205 | 2.456 |
| fast-marching (24-stencil) | N=385 | 9.9139 | 10.184 |
| lattice-basis-reduction FMM (fast-swept) | N=97 | 9.9377 | 5.847 |
| lattice-basis-reduction FMM (fast-swept) | N=193 | 9.9941 | 60.722 |

## Realistic (Pierson--Moskowitz) benchmark

| mode | direct | straight | ground-len | lateral-dev | weak-drift max | time-reduction |
|---|---|---|---|---|---|---|
| flat | 11.3137 | 11.3137 | 11.3137 | 0.0000 | 0.0000 | -0.00% |
| surface | 11.3670 | 11.3928 | 11.3139 | 0.0229 | 0.0000 | -0.47% |
| current | 9.9059 | 10.3837 | 11.6900 | 1.3447 | 0.3320 | +12.44% |
| combined | 9.9426 | 10.4561 | 11.6919 | 1.3501 | 0.3330 | +12.12% |

## Reverse benchmark (start=(9,9) goal=(1,1))

| mode | direct | straight | ground-len | lateral-dev | weak-drift max | time-reduction |
|---|---|---|---|---|---|---|
| flat | 11.3137 | 11.3137 | 11.3137 | 0.0000 | 0.0000 | -0.00% |
| surface | 11.6050 | 11.6667 | 11.3177 | 0.0934 | 0.0000 | -2.57% |
| current | 11.7160 | 12.9567 | 11.9145 | 1.7389 | 0.3926 | -3.56% |
| combined | 11.9871 | 13.4352 | 11.9260 | 1.7488 | 0.3926 | -5.95% |
