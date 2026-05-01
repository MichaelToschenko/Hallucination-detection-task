"""
probe.py — Hallucination probe classifier.

Binary MLP that classifies feature vectors as truthful (0) or hallucinated (1).
Public methods (`fit`, `fit_hyperparameters`, `predict`, `predict_proba`)
preserve the contract used by `evaluate.py` and `solution.py`.

Hyperparameters are read from environment variables so a bash runner can sweep
ablation configurations without editing source. Defaults give a 2-hidden-layer
GELU MLP with dropout 0.3, weight decay 1e-4, mini-batch 64, Adam lr 1e-3 and
early stopping (patience 30, max 300 epochs). PCA can be enabled to compress
high-dim feature spaces.

Setting PROBE_LEGACY=1 restores the original full-batch single-hidden-layer
classifier (no dropout, no early stopping, 200 epochs), used as a control in
ablation A11.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


PROBE_HIDDEN = _env_int("PROBE_HIDDEN", 256)
PROBE_LAYERS = _env_int("PROBE_LAYERS", 2)
PROBE_DROPOUT = _env_float("PROBE_DROPOUT", 0.3)
PROBE_WD = _env_float("PROBE_WD", 1e-4)
PROBE_LR = _env_float("PROBE_LR", 1e-3)
PROBE_EPOCHS = _env_int("PROBE_EPOCHS", 300)
PROBE_BATCH = _env_int("PROBE_BATCH", 64)
PROBE_PATIENCE = _env_int("PROBE_PATIENCE", 30)
PROBE_PCA = _env_int("PROBE_PCA", 0)
PROBE_LEGACY = _env_bool("PROBE_LEGACY", "0")
PROBE_TYPE = os.environ.get("PROBE_TYPE", "mlp").strip().lower()  # mlp | linear
PROBE_VAL_FRAC = _env_float("PROBE_VAL_FRAC", 0.15)
PROBE_SEED = _env_int("PROBE_SEED", 42)


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        if PROBE_LEGACY:
            self._net = nn.Sequential(
                nn.Linear(input_dim, PROBE_HIDDEN),
                nn.ReLU(),
                nn.Linear(PROBE_HIDDEN, 1),
            )
            return

        if PROBE_TYPE == "linear":
            self._net = nn.Sequential(nn.Linear(input_dim, 1))
            return

        layers: list[nn.Module] = [nn.Linear(input_dim, PROBE_HIDDEN), nn.GELU(), nn.Dropout(PROBE_DROPOUT)]
        for _ in range(max(0, PROBE_LAYERS - 1)):
            layers.extend([nn.Linear(PROBE_HIDDEN, PROBE_HIDDEN), nn.GELU(), nn.Dropout(PROBE_DROPOUT)])
        layers.append(nn.Linear(PROBE_HIDDEN, 1))
        self._net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network has not been built yet. Call fit() before forward().")
        return self._net(x).squeeze(-1)

    def _preprocess(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            X = self._scaler.fit_transform(X)
            if PROBE_PCA > 0 and X.shape[1] > PROBE_PCA:
                n_components = min(PROBE_PCA, X.shape[0] - 1, X.shape[1])
                self._pca = PCA(n_components=n_components, random_state=PROBE_SEED)
                X = self._pca.fit_transform(X)
            else:
                self._pca = None
        else:
            X = self._scaler.transform(X)
            if self._pca is not None:
                X = self._pca.transform(X)
        return X

    def _legacy_fit(self, X_t: torch.Tensor, y_t: torch.Tensor, criterion: nn.Module) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=PROBE_LR)
        self.train()
        for _ in range(200):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        self.eval()

    def _modern_fit(self, X_t: torch.Tensor, y_t: torch.Tensor, criterion: nn.Module) -> None:
        n = X_t.size(0)
        # Internal validation split for early stopping. Stratified on y.
        y_np = y_t.numpy().astype(int)
        # If a class has fewer than 2 samples we can't stratify — fall back to plain split.
        unique, counts = np.unique(y_np, return_counts=True)
        stratify = y_np if (len(unique) > 1 and counts.min() >= 2) else None
        idx = np.arange(n)
        idx_tr, idx_va = train_test_split(
            idx,
            test_size=PROBE_VAL_FRAC,
            random_state=PROBE_SEED,
            stratify=stratify,
        )
        X_tr, y_tr = X_t[idx_tr], y_t[idx_tr]
        X_va, y_va = X_t[idx_va], y_t[idx_va]

        optimizer = torch.optim.Adam(
            self.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD
        )

        n_tr = X_tr.size(0)
        batch_size = min(PROBE_BATCH, n_tr)

        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        rng = torch.Generator().manual_seed(PROBE_SEED)
        for _ in range(PROBE_EPOCHS):
            perm = torch.randperm(n_tr, generator=rng)
            self.train()
            for start in range(0, n_tr, batch_size):
                batch_idx = perm[start : start + batch_size]
                xb, yb = X_tr[batch_idx], y_tr[batch_idx]
                optimizer.zero_grad()
                loss = criterion(self(xb), yb)
                loss.backward()
                optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss = criterion(self(X_va), y_va).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PROBE_PATIENCE:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_pp = self._preprocess(X, fit=True)
        self._build_network(X_pp.shape[1])

        X_t = torch.from_numpy(X_pp).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if PROBE_LEGACY:
            self._legacy_fit(X_t, y_t, criterion)
        else:
            self._modern_fit(X_t, y_t, criterion)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_pp = self._preprocess(X, fit=False)
        X_t = torch.from_numpy(X_pp).float()
        self.eval()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
