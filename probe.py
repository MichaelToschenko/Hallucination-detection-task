from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


HIDDEN = 256
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 300
BATCH = 64
PATIENCE = 30
VAL_FRAC = 0.15
BASE_SEED = 42
ENSEMBLE_SIZE = 5  # train 5 networks with seeds 42..46 and average their predictions


def _build_mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, HIDDEN),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, 1),
    )


class HallucinationProbe(nn.Module):
    """Binary probe over a feature vector. Trains an ensemble of 5 MLPs."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._nets: list[nn.Sequential] = []
        self._threshold: float = 0.5

    # --- training ----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_s = self._scaler.fit_transform(X)
        X_t = torch.from_numpy(X_s).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # class imbalance: weight the positive class
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._nets = []
        for k in range(ENSEMBLE_SIZE):
            seed = BASE_SEED + k
            torch.manual_seed(seed)
            net = _build_mlp(X_s.shape[1])
            self._train_one(net, X_t, y_t, criterion, seed)
            self._nets.append(net)
        return self

    def _train_one(
        self,
        net: nn.Sequential,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        criterion: nn.Module,
        seed: int,
    ) -> None:
        # internal stratified split for early stopping
        y_np = y_t.numpy().astype(int)
        unique, counts = np.unique(y_np, return_counts=True)
        stratify = y_np if (len(unique) > 1 and counts.min() >= 2) else None
        idx_tr, idx_va = train_test_split(
            np.arange(X_t.size(0)),
            test_size=VAL_FRAC,
            random_state=seed,
            stratify=stratify,
        )
        X_tr, y_tr = X_t[idx_tr], y_t[idx_tr]
        X_va, y_va = X_t[idx_va], y_t[idx_va]

        optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        rng = torch.Generator().manual_seed(seed)
        batch = min(BATCH, X_tr.size(0))

        best_loss = float("inf")
        best_state: dict | None = None
        bad_epochs = 0

        for _ in range(EPOCHS):
            net.train()
            perm = torch.randperm(X_tr.size(0), generator=rng)
            for start in range(0, X_tr.size(0), batch):
                bi = perm[start : start + batch]
                optimizer.zero_grad()
                logits = net(X_tr[bi]).squeeze(-1)
                loss = criterion(logits, y_tr[bi])
                loss.backward()
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = criterion(net(X_va).squeeze(-1), y_va).item()

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        # tune the decision threshold on the validation split (F1 metric)
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_t = 0.5
        best_score = -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)
        self._threshold = best_t
        return self

    # --- inference ---------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._nets:
            raise RuntimeError("Probe is not fitted yet.")
        X_t = torch.from_numpy(self._scaler.transform(X)).float()

        probs: list[np.ndarray] = []
        with torch.no_grad():
            for net in self._nets:
                net.eval()
                logits = net(X_t).squeeze(-1)
                probs.append(torch.sigmoid(logits).numpy())

        prob_pos = np.mean(probs, axis=0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
