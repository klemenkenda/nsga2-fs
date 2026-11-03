# API

Everything is in the module `nsgaii_miip`. Gene vectors are NumPy boolean
arrays, one entry per feature; populations are 2-D boolean arrays, one row per
individual. Functions that draw randomness take a `numpy.random.RandomState`.

## Drivers

### `run_nsga2(n_features, objective, seed, stop_evaluations, ...)`

Plain NSGA-II, with no clustering or mutual information.

| argument | default | meaning |
|---|---|---|
| `n_features` | | number of candidate features |
| `objective` | | `genes -> float`, the score to maximise |
| `seed` | | random seed |
| `stop_evaluations` | | fit budget (new subsets scored) |
| `pop_size` | 100 | population size |
| `c_rate` | 0.9 | crossover probability per pair |
| `m_rate` | `1 / n_features` | base mutation rate |
| `init` | `"obl"` | start: `"obl"`, `"sparse"`, `"uniform"` or `"loguniform"` (see `INITIALIZATIONS`) |
| `mutation` | `"balanced"` | `"balanced"` or `"min1"` |
| `rng_streams` | False | separate random streams for the start and the operators, so runs that differ only in operators start from the same population |
| `max_barren_generations` | None | stop after this many generations in a row with no new subset; None never stops on this |
| `stop_at_size` | None | stop after the generation in which a subset of at most this many features was first scored |
| `progress_cb` | None | `evaluations -> None`, at the start of each generation |
| `iteration_cb` | None | `(iteration, evaluations) -> None`, after the start (0) and after every generation |
| `scored_cb` | None | `(genes, score, seconds) -> None`, after every new fit |

Returns a dict:

| key | |
|---|---|
| `pareto_genes` | boolean array, the final non-dominated subsets |
| `pareto_objs` | float array `[-score, size]` per subset |
| `evaluations` | fits spent |
| `iterations` | generations run |
| `unique_subsets_cached` | distinct subsets scored |
| `start_population_sha256` | first 16 hex digits of the hash of the start population |
| `stopped_on` | `"budget"`, `"stagnation"` or `"reach"` |

### `run_nsgaii_miip(n_features, objective, clusters, mi, ent, seed, stop_evaluations, ...)`

NSGAII-MIIP (`NSGAIIMIIP.m`). `clusters`, `mi` and `ent` come from
`cluster_features`. Each generation applies the MIIP improvement phase to the
non-dominated set of parents and offspring before survival. The arguments
`pop_size`, `c_rate`, `m_rate`, `progress_cb`, `iteration_cb` and
`max_barren_generations` work as in `run_nsga2`. `init` is `"obl"` (the
source) or `"sparse"`, and `verbose` prints each generation. It returns the
same keys as `run_nsga2` except `start_population_sha256`, and `stopped_on` is
never `"reach"`.

## Information and clustering

* `cluster_features(X, y, rng, n_jobs=1) -> (clusters, mi, ent)`: discretises
  `X`, builds the pairwise MI matrix with the class appended as the last
  row/column, and clusters the features into about `sqrt(n_features)` groups
  by k-medoids on `1 - MI / min(H_i, H_j)`. Its cost is O(F^2) in the number
  of features.
* `discretize(X, n_bins=N_BINS)`: equal-width binning per column (Weka's
  default, 10 bins).
* `entropy_vector(codes)`, `mi_matrix(codes, n_jobs=1)`: entropies and
  pairwise mutual information, in bits, of discretised columns.
* `kmedoids_on_distance(D, k, rng, max_iter=100)`: k-medoids labels for a
  precomputed distance matrix.
* `improvement(pop, mi, ent, clusters, rng)`: the MIIP phase. It returns three
  variants per solution (forward, backward, interchange).

## Operators

* `fast_nondominated_sort(objs) -> list of index arrays`, minimising every
  column.
* `crowding_distance(objs)`
* `binary_tournament_crossover(pop, rank, crowd, c_rate, rng)`
* `balanced_mutation(pop, m_rate, rng)`: 1-bits flip at `m_rate` and 0-bits at
  `m_rate * ones / zeros`, so additions and deletions balance on average.
* `balanced_mutation_min1(pop, m_rate, rng) -> (pop, forced)`: the same, plus
  one forced flip in every child left unchanged.
* Starts: `obl_initialization`, `sparse_initialization(..., max_size=30)`,
  `uniform_initialization`, `loguniform_initialization`, all called as
  `(pop_size, n_features, rng)`.
* `barren_counter(limit)`: the stagnation rule as a callable fed the fit count
  after each generation.

## Constants

* `N_BINS = 10`
* `MAX_BARREN_GENERATIONS = 50`, a recommended stagnation limit. The drivers
  default to None.
* `INITIALIZATIONS`, the starts by name.
