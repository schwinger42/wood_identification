import numpy as np
import pandas as pd

from woodraman.models import PLSDA
from woodraman.preprocess import NuisanceProjector, session_difference_basis


def test_projector_removes_basis_direction():
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    v /= np.linalg.norm(v)
    X = rng.normal(size=(20, 50)) + 5 * rng.normal(size=(20, 1)) * v
    Z = NuisanceProjector(v[None, :]).transform(X)
    assert np.allclose(Z @ v, 0, atol=1e-10)


def test_session_basis_empty_without_repeats():
    meta = pd.DataFrame({"block_id": ["a", "b"], "face": ["C", "C"], "date": ["d1", "d2"]})
    assert session_difference_basis(np.ones((2, 5)), meta).shape == (0, 5)


def test_plsda_separates_easy_classes():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (30, 10)), rng.normal(3, 1, (30, 10))]
    y = np.array(["a"] * 30 + ["b"] * 30)
    assert (PLSDA(2).fit(X, y).predict(X) == y).mean() > 0.95
