from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


N_FOLDS = 10
VAL_FRAC = 0.15
SEED = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a list of (idx_train, idx_val, idx_test) tuples, one per fold."""
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, (non_test_pos, test_pos) in enumerate(skf.split(idx, y)):
        idx_test = idx[test_pos]
        idx_non_test = idx[non_test_pos]
        idx_train, idx_val = train_test_split(
            idx_non_test,
            test_size=VAL_FRAC,
            random_state=SEED + fold_idx,
            stratify=y[idx_non_test],
        )
        folds.append((idx_train, idx_val, idx_test))
    return folds
