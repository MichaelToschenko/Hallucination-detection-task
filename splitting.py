"""
splitting.py — Train / validation / test split utilities.

`split_data` returns a list of `(idx_train, idx_val, idx_test)` tuples. With
the default `SPLIT_KFOLDS=1` it produces a single stratified 0.7/0.15/0.15
split. With `SPLIT_KFOLDS=K` (K>=2) it produces K stratified folds: each fold
holds out one chunk as test, splits the rest into train/val (15% val, stratified).

Contract preserved:
* Indices are 1-D NumPy arrays of integers.
* `idx_val` may be None.
* Within a single tuple, indices are non-overlapping.
* Across folds, every sample appears exactly once in `idx_test`.

Environment variable:
    SPLIT_KFOLDS    int — 1 = single split, K>=2 = K-fold (default 1).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


SPLIT_KFOLDS: int = int(os.environ.get("SPLIT_KFOLDS", "1"))


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets."""
    idx = np.arange(len(y))

    if SPLIT_KFOLDS <= 1:
        idx_train_val, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        relative_val = val_size / (1.0 - test_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        return [(idx_train, idx_val, idx_test)]

    skf = StratifiedKFold(n_splits=SPLIT_KFOLDS, shuffle=True, random_state=random_state)
    folds: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (non_test_pos, test_pos) in enumerate(skf.split(idx, y)):
        idx_test = idx[test_pos]
        idx_non_test = idx[non_test_pos]
        idx_train, idx_val = train_test_split(
            idx_non_test,
            test_size=val_size,
            random_state=random_state + fold_idx,
            stratify=y[idx_non_test],
        )
        folds.append((idx_train, idx_val, idx_test))
    return folds
