# Notes on the port

Source: <https://github.com/andali89/NSGAII-MIIP> (MATLAB + Weka, MIT,
(c) 2024 An-Da Li). Paper: Li et al., EJOR 2025.

## What maps to what

| Python | MATLAB source |
|---|---|
| `run_nsgaii_miip` | `NSGAIIMIIP.m` |
| `cluster_features`, `discretize`, `mi_matrix`, `entropy_vector` | `clusterFeature.m`, `getEntropyMatrix`; Weka's unsupervised `Discretize` (10 equal-width bins; the source's `setBins` call is commented out) |
| `kmedoids_on_distance` | `kmedoids.m` (a Voronoi-iteration k-medoids stands in for it) |
| `improvement`, `_forward`, `_backward`, `_interchange`, `_weight` | `improvement.m`, `calWeight` |
| `obl_initialization` | `oblInitialization.m` |
| `binary_tournament_crossover` | `binTour.m` |
| `balanced_mutation` | `mutation.m` |
| `fast_nondominated_sort`, `crowding_distance`, survival | standard NSGA-II (Deb et al. 2002) |

## Departures inside the ported code

* **Objective.** The paper maximises the geometric mean over internal CV
  folds. Here the objective is whatever the caller passes. The size objective
  is unchanged.
* **Budget.** The paper's default is 10,000 evaluations. Here
  `stop_evaluations` is set by the caller and is exact: once it is spent, a
  subset that would need a new fit is dropped instead of scored.
* **Zero-redundancy guard.** `_weight` clamps a non-positive redundancy to
  1e-12. Without the clamp, an all-zero-MI bench divides by zero; the source
  has no guard.
* **Negative distances.** Normalised MI can exceed 1 by a rounding error, so
  distances are clipped at 0 before k-medoids.
* **Empty subsets** get the worst score and are never fit.

## Added, not in the source

The source does not have any of these. With the defaults (`init="obl"`,
`mutation="balanced"`, no stagnation or reach stop) the search is the ported
one.

* `run_nsga2`: the same search without the improvement phase.
* `sparse_initialization`, `uniform_initialization`,
  `loguniform_initialization`: the OBL start sits at about 50% density, which
  on hundreds of features begins far from small subsets. These starts cover
  small sizes or all sizes.
* `balanced_mutation_min1`: a forced flip for children that mutation left
  unchanged.
* `barren_counter` / `max_barren_generations`: a stagnation stop. Without it a
  converged population re-proposes cached subsets that cost no fits, and the
  loop never reaches the budget.
* `stop_at_size`, `rng_streams` and the callbacks in `run_nsga2`.

The comments in `nsgaii_miip.py` cite FASTENER experiments (H5H6-E01, H14,
DEC-41 and so on), the studies these options were added for, in
[fastener-experiments](https://github.com/klemenkenda/fastener-experiments).
