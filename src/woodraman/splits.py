"""Group-aware splitting. Grouping is by block_id or tree_id, never by spot."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

LEVELS = ("tree_id", "block_id")


def leave_one_group_out(meta: pd.DataFrame, level: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx), holding out one group at `level` per fold."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    groups = meta[level].to_numpy()
    for g in pd.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        yield train, test


def held_out_tree_folds(meta: pd.DataFrame) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """For every species with >1 tree, hold out each of its trees in turn (all other data trains).

    This is the only honest species-generalisation test the current data allows.
    """
    trees_per_species = meta.groupby("species")["tree_id"].nunique()
    for sp in trees_per_species[trees_per_species > 1].index:
        for tree in sorted(meta.loc[meta.species == sp, "tree_id"].unique()):
            test = np.flatnonzero(meta.tree_id.to_numpy() == tree)
            train = np.flatnonzero(meta.tree_id.to_numpy() != tree)
            yield tree, train, test


def assert_no_leak(meta: pd.DataFrame, train: np.ndarray, test: np.ndarray, level: str) -> None:
    """Raise if any group at `level` appears on both sides of the split."""
    overlap = set(meta.iloc[train][level]) & set(meta.iloc[test][level])
    if overlap:
        raise AssertionError(f"Leak at {level}: {sorted(overlap)[:5]}")
