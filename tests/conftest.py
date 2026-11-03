import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def toy():
    """40 features, the first 4 informative; the objective is the share of
    selected features that are informative minus a small size penalty, so it
    is cheap and deterministic."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(300, 40))
    y = (X[:, :4].sum(axis=1) > 0).astype(int)

    def objective(genes):
        f = np.where(genes)[0]
        return float(np.isin(f, range(4)).sum()) / 4 - 0.01 * f.size
    return X, y, objective
