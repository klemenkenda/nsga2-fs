# nsga2-fs

NSGA-II for feature selection, in one dependency-light Python module
(`nsgaii_miip.py`, needs only NumPy). It can be run two ways:

* **plain NSGA-II** (`run_nsga2`): a bi-objective wrapper search that maximises
  a score you supply and minimises the number of selected features;
* **NSGAII-MIIP** (`run_nsgaii_miip`): the same search plus the
  mutual-information-guided improvement phase of Li et al., ported from the
  authors' MATLAB source.

The operators are those of NSGAII-MIIP: opposition-based initialisation,
binary tournament on (front, crowding), single-point crossover at a differing
locus, balanced mutation, NSGA-II survival. Further starts (sparse, uniform and
log-uniform over sizes) and a mutation that never leaves a child unchanged
(`min1`) are included as options.

> **This is a reimplementation, not the authors' code or results.** The
> original runs on MATLAB and Weka. Every operator is ported from the
> published source, but a port can differ in ways that only show up in the
> numbers. See [docs/PORT.md](docs/PORT.md).

## Install

```bash
pip install git+https://github.com/klemenkenda/nsga2-fs
```

or clone it and put the folder on `sys.path`; it is a single flat module,
`import nsgaii_miip`. `joblib` is optional and only speeds up the
mutual-information matrix (`n_jobs`).

## Quick start

The objective takes a boolean gene vector (one entry per feature) and returns
the score to **maximise**, typically a validation score of a model fit on the
selected columns:

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import nsgaii_miip as nm

def objective(genes):
    f = np.where(genes)[0]
    model = DecisionTreeClassifier(random_state=0).fit(X_train[:, f], y_train)
    return f1_score(y_val, model.predict(X_val[:, f]), average="weighted")

# plain NSGA-II
res = nm.run_nsga2(X_train.shape[1], objective, seed=0, stop_evaluations=6000)

# NSGAII-MIIP: cluster the features once, then search
clusters, mi, ent = nm.cluster_features(X_train, y_train, np.random.RandomState(0))
res = nm.run_nsgaii_miip(X_train.shape[1], objective, clusters, mi, ent,
                         seed=0, stop_evaluations=6000)

for genes, (neg_score, size) in zip(res["pareto_genes"], res["pareto_objs"]):
    print(int(size), -neg_score, np.where(genes)[0])
```

Both return the final Pareto front: `pareto_genes` (boolean, one row per
subset) and `pareto_objs` (`[-score, size]`, as NSGA-II minimises both).

## Budget and stopping

`stop_evaluations` counts **fits**, that is, calls to `objective` for a subset
not seen before. Repeats are cached and cost nothing, and the budget is exact:
the last generation is cut short rather than overshooting. A converged
population can keep re-proposing cached subsets without spending budget, so
pass `max_barren_generations` (`MAX_BARREN_GENERATIONS = 50` is the value used
in practice) to stop after that many generations with no new subset.
`run_nsga2` can also stop once a subset of at most `stop_at_size` features has
been scored. `stopped_on` in the result says which stop fired.

## Documentation

* [docs/API.md](docs/API.md): every public function, its arguments and return
  values.
* [docs/PORT.md](docs/PORT.md): where each function comes from in the MATLAB
  source, what was changed and what was added.

## Tests

```bash
pip install pytest scikit-learn
pytest tests
```

## Provenance and citation

Written for the FASTENER experiments
([klemenkenda/fastener-experiments](https://github.com/klemenkenda/fastener-experiments)),
where it is the `nsga2`, `nsga2_sparse` and `nsgaii_miip` comparators. The
first commit of this repository is byte-identical to the file those
experiments ran.

If you use the MIIP phase, cite the method:

> A.-D. Li et al., "A multi-objective evolutionary algorithm with
> mutual-information-guided improvement phase for feature selection in complex
> manufacturing processes", *European Journal of Operational Research*, 2025.
> Source: <https://github.com/andali89/NSGAII-MIIP>

and for NSGA-II itself: K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast
and elitist multiobjective genetic algorithm: NSGA-II", *IEEE Transactions on
Evolutionary Computation* 6(2), 182-197, 2002.

## Licence

MIT. The copyright of the MATLAB source (An-Da Li, 2024) is kept alongside
that of the port; see [LICENSE](LICENSE).
