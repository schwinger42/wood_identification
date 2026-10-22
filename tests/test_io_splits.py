import numpy as np
import pytest

from woodraman.io import RAW_DIR, load_dataset, parse_filename
from woodraman.splits import assert_no_leak, held_out_tree_folds, leave_one_group_out


@pytest.mark.parametrize(
    "name, species, tree, block, face, spot, date, power",
    [
        (
            "24mW-20250925-Chamaecyparis obtusa var. formosana-2-4-R-8.txt",
            "Chamaecyparis obtusa var. formosana",
            2,
            4,
            "R",
            8,
            "20250925",
            24,
        ),
        ("24mW-20250926-Swietenia mahagoni-3-T-10.txt", "Swietenia mahagoni", 1, 3, "T", 10, "20250926", 24),
        ("20250923-Fokienia hodginsii-3-T-11.txt", "Fokienia hodginsii", 1, 3, "T", 11, "20250923", None),
        (
            "24mW-20250926-Swietenia marcrophylla-2-1-C-9.txt",
            "Swietenia macrophylla",
            2,
            1,
            "C",
            9,
            "20250926",
            24,
        ),
        ("20250922-Chamaecyparis obyusa-4-C-1.txt", "Chamaecyparis obtusa", 1, 4, "C", 1, "20250922", None),
    ],
)
def test_parse_filename(name, species, tree, block, face, spot, date, power):
    m = parse_filename(name)
    assert (m.species, m.tree, m.block, m.face, m.spot, m.date, m.power_mw) == (
        species,
        tree,
        block,
        face,
        spot,
        date,
        power,
    )


def test_parse_rejects_unknown():
    with pytest.raises(ValueError):
        parse_filename("not-a-spectrum.txt")


@pytest.fixture(scope="module")
def data():
    if not RAW_DIR.exists():
        pytest.skip("dataset/raw not present")
    return load_dataset()


def test_dataset_shape(data):
    X, wn, meta = data
    assert X.shape == (1050, 2005)
    assert meta.species.nunique() == 9
    assert meta.tree_id.nunique() == 11
    assert np.all(np.diff(wn) > 0)


@pytest.mark.parametrize("level", ["block_id", "tree_id"])
def test_logo_never_leaks(data, level):
    _, _, meta = data
    for train, test in leave_one_group_out(meta, level):
        assert_no_leak(meta, train, test, level)
        assert_no_leak(meta, train, test, "block_id")


def test_held_out_tree_folds(data):
    _, _, meta = data
    trees = [t for t, _, _ in held_out_tree_folds(meta)]
    assert len(trees) == 4  # 2 species x 2 trees
    for _, train, test in held_out_tree_folds(meta):
        assert_no_leak(meta, train, test, "tree_id")
