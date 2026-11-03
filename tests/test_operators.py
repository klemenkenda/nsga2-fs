import numpy as np

import nsgaii_miip as nm


def test_discretize_bins_and_constant_column():
    X = np.c_[np.linspace(0, 1, 101), np.ones(101)]
    codes = nm.discretize(X)
    assert codes[:, 0].min() == 0 and codes[:, 0].max() == nm.N_BINS - 1
    assert (codes[:, 1] == 0).all()


def test_mi_matrix_symmetric_diagonal_is_entropy():
    codes = np.random.RandomState(1).randint(0, 4, size=(500, 5)).astype(np.int32)
    M = nm.mi_matrix(codes)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), nm.entropy_vector(codes))


def test_cluster_features_covers_every_feature_once(toy):
    X, y, _ = toy
    clusters, mi, ent = nm.cluster_features(X, y, np.random.RandomState(0))
    assert sorted(np.concatenate(clusters).tolist()) == list(range(X.shape[1]))
    assert mi.shape == (X.shape[1] + 1,) * 2 and ent.shape == (X.shape[1] + 1,)


def test_nondominated_sort_and_crowding():
    objs = np.array([[0, 3], [1, 1], [3, 0], [2, 2], [3, 3]], dtype=float)
    fronts = nm.fast_nondominated_sort(objs)
    assert sorted(fronts[0].tolist()) == [0, 1, 2]
    assert fronts[1].tolist() == [3] and fronts[2].tolist() == [4]
    d = nm.crowding_distance(objs[fronts[0]])
    assert np.isinf(d).sum() == 2 and np.isfinite(d).sum() == 1


def test_initializations_shapes_and_sizes():
    rng = np.random.RandomState(0)
    obl = nm.obl_initialization(10, 50, rng)
    assert obl.shape == (10, 50)
    assert (obl[:5] ^ obl[5:]).all()          # opposition pairs
    sparse = nm.sparse_initialization(20, 50, rng, max_size=5)
    assert 1 <= sparse.sum(1).min() and sparse.sum(1).max() <= 5
    for init in (nm.uniform_initialization, nm.loguniform_initialization):
        pop = init(50, 30, rng)
        assert pop.shape == (50, 30) and pop.sum(1).min() >= 1


def test_balanced_mutation_keeps_expected_size():
    rng = np.random.RandomState(0)
    pop = np.zeros((4000, 100), dtype=bool)
    pop[:, :10] = True
    out = nm.balanced_mutation(pop, 0.05, rng)
    assert abs(out.sum(1).mean() - 10) < 0.1


def test_min1_changes_every_child():
    rng = np.random.RandomState(0)
    pop = nm.uniform_initialization(200, 20, rng)
    out, forced = nm.balanced_mutation_min1(pop, 1e-9, rng)
    assert forced.all() and (out != pop).any(axis=1).all()


def test_barren_counter():
    step = nm.barren_counter(2)
    assert [step(e) for e in (5, 5, 5)] == [False, False, True]
    assert not any(nm.barren_counter(None)(5) for _ in range(10))
