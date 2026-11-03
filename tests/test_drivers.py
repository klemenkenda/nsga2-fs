import numpy as np
import pytest

import nsgaii_miip as nm


def front_sets(result):
    return sorted(tuple(np.where(g)[0]) for g in result["pareto_genes"])


@pytest.mark.parametrize("init", sorted(nm.INITIALIZATIONS))
def test_nsga2_budget_is_exact_and_deterministic(toy, init):
    _, _, objective = toy
    a = nm.run_nsga2(40, objective, seed=7, stop_evaluations=300, pop_size=20, init=init)
    b = nm.run_nsga2(40, objective, seed=7, stop_evaluations=300, pop_size=20, init=init)
    assert a["evaluations"] == 300 and a["stopped_on"] == "budget"
    assert front_sets(a) == front_sets(b)


def test_nsga2_finds_the_informative_features(toy):
    _, _, objective = toy
    r = nm.run_nsga2(40, objective, seed=0, stop_evaluations=3000, pop_size=40,
                     init="sparse")
    assert (0, 1, 2, 3) in front_sets(r)


def test_nsga2_stops_on_reach_and_on_stagnation(toy):
    _, _, objective = toy
    r = nm.run_nsga2(40, objective, seed=0, stop_evaluations=5000, stop_at_size=2)
    assert r["stopped_on"] == "reach" and r["evaluations"] < 5000
    r = nm.run_nsga2(40, objective, seed=0, stop_evaluations=10 ** 6,
                     pop_size=10, init="sparse", max_barren_generations=3)
    assert r["stopped_on"] == "stagnation"


def test_nsga2_rng_streams_share_the_start(toy):
    _, _, objective = toy
    runs = [nm.run_nsga2(40, objective, seed=3, stop_evaluations=200, init="uniform",
                         mutation=m, rng_streams=True) for m in ("balanced", "min1")]
    assert runs[0]["start_population_sha256"] == runs[1]["start_population_sha256"]


def test_nsga2_callbacks(toy):
    _, _, objective = toy
    scored, its = [], []
    r = nm.run_nsga2(40, objective, seed=0, stop_evaluations=150, pop_size=20,
                     scored_cb=lambda g, s, t: scored.append(s),
                     iteration_cb=lambda i, e: its.append(i))
    assert len(scored) == r["evaluations"] and its[0] == 0 and its[-1] == r["iterations"]


def test_nsgaii_miip_runs_and_is_deterministic(toy):
    X, y, objective = toy
    clusters, mi, ent = nm.cluster_features(X, y, np.random.RandomState(0))
    args = (40, objective, clusters, mi, ent)
    a = nm.run_nsgaii_miip(*args, seed=1, stop_evaluations=400, pop_size=20)
    b = nm.run_nsgaii_miip(*args, seed=1, stop_evaluations=400, pop_size=20)
    assert a["evaluations"] == 400 and front_sets(a) == front_sets(b)


def test_unknown_options_raise(toy):
    _, _, objective = toy
    with pytest.raises(ValueError):
        nm.run_nsga2(40, objective, seed=0, stop_evaluations=10, init="nope")
    with pytest.raises(ValueError):
        nm.run_nsga2(40, objective, seed=0, stop_evaluations=10, mutation="nope")
