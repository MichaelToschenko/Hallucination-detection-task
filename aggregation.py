"""
aggregation.py — Token aggregation strategy and feature extraction.

Two stages, both controlled via environment variables so a single bash runner
can sweep ablation configurations without editing source.

Stage 1 — `aggregate`: per layer in `LAYERS`, apply each pool in `POOLS`
    ("last", "mean", "max", "tail_mean", "tail_max") over the real (non-padded)
    tokens, concatenate. Tail pools operate on the last `AGG_TAIL_N` real tokens
    only — useful as a heuristic for response-only pooling without a tokenizer.

Stage 2 — `extract_geometric_features`: hand-crafted statistics over hidden
    states (norms, representation drift, last-vs-mean, sequence length,
    optional spectral / topological signals). Each group toggles independently.

Defaults reproduce the recommended A4 configuration:
    LAYERS=-4,-3,-2,-1   POOLS=last,mean   GEOM_*=1   GEOM_TOPOLOGY=0
yielding feature_dim = 4*2*896 + 225 = 7393.

Environment variables (all optional):
    AGG_LAYERS          comma-separated layer indices (negative ok)
    AGG_POOLS           comma-separated pool names from
                        {last, mean, max, tail_mean, tail_max}
    AGG_TAIL_N          int — window size for tail_* pools (default 16)
    GEOM_NORMS          1/0 — G1+G2 layer-wise L2 norms (50 dims)
    GEOM_DRIFT          1/0 — G3+G4 inter-layer cos / L2 (48 dims)
    GEOM_TOKEN_STATS    1/0 — G5 per-layer mean/std/max of token L2 norms (75)
    GEOM_LAST_VS_MEAN   1/0 — G6+G7 cos / L2 last-vs-mean per layer (50)
    GEOM_SEQ_LEN        1/0 — G8 normalised sequence length (1)
    GEOM_GLOBAL_DRIFT   1/0 — G9 cos(emb-mean, last-layer-mean) (1)
    GEOM_TOPOLOGY       1/0 — G10..G12 SVD + intra-layer pdist + entropy (12)
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


# Sequence length used for normalising G8. Duplicates MAX_LENGTH from model.py
# so this module stays import-independent of the fixed-infrastructure files.
_MAX_LENGTH = 512


def _env_layers(name: str, default: str) -> tuple[int, ...]:
    raw = os.environ.get(name, default)
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _env_pools(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


LAYERS: tuple[int, ...] = _env_layers("AGG_LAYERS", "-4,-3,-2,-1")
POOLS: tuple[str, ...] = _env_pools("AGG_POOLS", "last,mean")
AGG_TAIL_N: int = int(os.environ.get("AGG_TAIL_N", "16"))
GEOM_NORMS: bool = _env_bool("GEOM_NORMS", "1")
GEOM_DRIFT: bool = _env_bool("GEOM_DRIFT", "1")
GEOM_TOKEN_STATS: bool = _env_bool("GEOM_TOKEN_STATS", "1")
GEOM_LAST_VS_MEAN: bool = _env_bool("GEOM_LAST_VS_MEAN", "1")
GEOM_SEQ_LEN: bool = _env_bool("GEOM_SEQ_LEN", "1")
GEOM_GLOBAL_DRIFT: bool = _env_bool("GEOM_GLOBAL_DRIFT", "1")
GEOM_TOPOLOGY: bool = _env_bool("GEOM_TOPOLOGY", "0")


def _pool_layer(
    layer: torch.Tensor,
    mask: torch.Tensor,
    n_real: int,
    last_pos: int,
    real_idx: torch.Tensor,
    pools: tuple[str, ...],
) -> list[torch.Tensor]:
    """Return pooled vectors for one layer, in the order given by `pools`."""
    parts: list[torch.Tensor] = []
    for pool in pools:
        if pool == "last":
            parts.append(layer[last_pos])
        elif pool == "mean":
            parts.append((layer * mask.unsqueeze(-1)).sum(0) / n_real)
        elif pool == "max":
            masked = layer.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            parts.append(masked.max(0).values)
        elif pool in ("tail_mean", "tail_max"):
            n = min(AGG_TAIL_N, real_idx.numel())
            tail_idx = real_idx[-n:]
            sub = layer.index_select(0, tail_idx)
            parts.append(sub.mean(0) if pool == "tail_mean" else sub.max(0).values)
        else:
            raise ValueError(
                f"Unknown pool '{pool}'. Expected one of: "
                f"last, mean, max, tail_mean, tail_max."
            )
    return parts


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a feature vector.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,) with 1 for real tokens.

    Returns:
        1-D float32 tensor on CPU of shape (len(LAYERS) * len(POOLS) * hidden_dim,).
    """
    device = hidden_states.device
    mask = attention_mask.to(device=device, dtype=torch.bool)
    n_real = int(mask.sum().item())
    real_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_idx[-1].item())

    parts: list[torch.Tensor] = []
    for l in LAYERS:
        parts.extend(_pool_layer(hidden_states[l], mask, n_real, last_pos, real_idx, POOLS))

    return torch.cat(parts, dim=0).to(torch.float32).cpu()


def _layer_mean_pools(
    hidden_states: torch.Tensor, mask: torch.Tensor, n_real: int
) -> torch.Tensor:
    """Return (n_layers, hidden_dim) mean-pooled tensor over real tokens."""
    masked = hidden_states * mask.view(1, -1, 1)
    return masked.sum(dim=1) / n_real


def _layer_last_tokens(
    hidden_states: torch.Tensor, last_pos: int
) -> torch.Tensor:
    """Return (n_layers, hidden_dim) tensor with the last-real-token vector per layer."""
    return hidden_states[:, last_pos, :]


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Hand-crafted statistical / geometric features.

    Args:
        hidden_states:  Tensor of shape (n_layers, seq_len, hidden_dim).
        attention_mask: 1-D tensor of shape (seq_len,) with 1 for real tokens.

    Returns:
        1-D float32 tensor on CPU. Length depends on which GEOM_* flags are on.
        With defaults (all on except GEOM_TOPOLOGY): 225.
    """
    device = hidden_states.device
    mask = attention_mask.to(device=device, dtype=torch.bool)
    n_real = int(mask.sum().item())
    last_pos = int(mask.nonzero(as_tuple=False)[-1].item())
    n_layers = hidden_states.size(0)

    feats: list[torch.Tensor] = []

    mean_pools = _layer_mean_pools(hidden_states, mask, n_real)  # (L, D)
    last_tokens = _layer_last_tokens(hidden_states, last_pos)    # (L, D)

    if GEOM_NORMS:
        feats.append(mean_pools.norm(dim=-1))   # G1 (L,)
        feats.append(last_tokens.norm(dim=-1))  # G2 (L,)

    if GEOM_DRIFT:
        a = mean_pools[:-1]
        b = mean_pools[1:]
        feats.append(F.cosine_similarity(a, b, dim=-1, eps=1e-8))  # G3 (L-1,)
        feats.append((b - a).norm(dim=-1))                          # G4 (L-1,)

    if GEOM_TOKEN_STATS:
        # G5: per-layer mean/std/max of per-token L2 norms over real tokens.
        token_norms = hidden_states.norm(dim=-1)  # (L, S)
        # For std/mean/max over real tokens only — gather valid columns.
        real_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        valid_norms = token_norms.index_select(1, real_idx)  # (L, n_real)
        feats.append(valid_norms.mean(dim=-1))
        feats.append(
            valid_norms.std(dim=-1, unbiased=False)
            if n_real > 1
            else torch.zeros(n_layers, device=device)
        )
        feats.append(valid_norms.max(dim=-1).values)

    if GEOM_LAST_VS_MEAN:
        feats.append(F.cosine_similarity(last_tokens, mean_pools, dim=-1, eps=1e-8))  # G6
        feats.append((last_tokens - mean_pools).norm(dim=-1))                          # G7

    if GEOM_SEQ_LEN:
        feats.append(torch.tensor([n_real / _MAX_LENGTH], device=device))  # G8

    if GEOM_GLOBAL_DRIFT:
        # cos(embedding-layer mean, last-transformer-layer mean) — global drift.
        feats.append(
            F.cosine_similarity(
                mean_pools[0].unsqueeze(0),
                mean_pools[-1].unsqueeze(0),
                dim=-1,
                eps=1e-8,
            )
        )  # G9 (1,)

    if GEOM_TOPOLOGY:
        real_idx = mask.nonzero(as_tuple=False).squeeze(-1)

        # G10: top-3 singular values + sum of squares from the last layer's
        # token matrix.
        last_layer_real = hidden_states[-1].index_select(0, real_idx)  # (n_real, D)
        try:
            sv = torch.linalg.svdvals(last_layer_real)
            top = torch.zeros(3, device=device)
            top[: min(3, sv.numel())] = sv[: min(3, sv.numel())]
            sum_sq = (sv ** 2).sum().unsqueeze(0)
            feats.append(torch.cat([top, sum_sq]))  # (4,)
        except Exception:
            feats.append(torch.zeros(4, device=device))

        # G11: mean pairwise L2 distance on a 32-token subsample, last 4 layers.
        subset_layers = list(LAYERS) if len(LAYERS) >= 1 else [-1]
        # Use the last 4 of LAYERS (or fewer, padded with the last layer) so
        # G11/G12 always emit exactly 4 values.
        target_layers = list(LAYERS[-4:]) if len(LAYERS) >= 4 else list(LAYERS) + [-1] * (4 - len(LAYERS))
        sample_n = min(32, n_real)
        # Deterministic subsample: linearly spaced indices across real tokens.
        if sample_n >= 2:
            stride = max(1, n_real // sample_n)
            sub_idx = real_idx[:: stride][:sample_n]
        else:
            sub_idx = real_idx
        pdist_vals = []
        for l in target_layers:
            sub = hidden_states[l].index_select(0, sub_idx)  # (k, D)
            if sub.size(0) >= 2:
                d = torch.cdist(sub.unsqueeze(0), sub.unsqueeze(0)).squeeze(0)
                # Mean of upper triangle (off-diagonal).
                k = sub.size(0)
                triu_mask = torch.triu(torch.ones(k, k, device=device, dtype=torch.bool), diagonal=1)
                pdist_vals.append(d[triu_mask].mean().unsqueeze(0))
            else:
                pdist_vals.append(torch.zeros(1, device=device))
        feats.append(torch.cat(pdist_vals))  # (4,)

        # G12: entropy of normalised token L2 norms in the same 4 layers.
        ent_vals = []
        for l in target_layers:
            norms = hidden_states[l].index_select(0, real_idx).norm(dim=-1)  # (n_real,)
            total = norms.sum()
            if total > 0:
                p = norms / total
                ent = -(p * torch.log(p.clamp_min(1e-12))).sum()
            else:
                ent = torch.zeros((), device=device)
            ent_vals.append(ent.unsqueeze(0))
        feats.append(torch.cat(ent_vals))  # (4,)

    if not feats:
        return torch.zeros(0, dtype=torch.float32)

    return torch.cat([f.flatten() for f in feats], dim=0).to(torch.float32).cpu()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Concatenates `aggregate(...)` with `extract_geometric_features(...)` when
    `use_geometric=True`.
    """
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        if geo_features.numel() > 0:
            return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
