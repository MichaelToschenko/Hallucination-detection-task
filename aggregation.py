from __future__ import annotations

import torch


LAYERS = (-14, -12, -10)  # negative indices into hidden_states 
TAIL_N = 16               # window size for tail_mean pool


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool hidden states into a (5376,) feature vector.

    Args:
        hidden_states:  (n_layers, seq_len, hidden_dim) tensor.
        attention_mask: (seq_len,) tensor with 1 for real tokens, 0 for padding.

    Returns:
        1-D float32 CPU tensor of length len(LAYERS) * 2 * hidden_dim.
    """
    device = hidden_states.device
    mask = attention_mask.to(device=device, dtype=torch.bool)
    real_idx = mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_idx[-1].item())

    # tail = last TAIL_N real tokens (or all of them if the sequence is shorter)
    n_tail = min(TAIL_N, real_idx.numel())
    tail_idx = real_idx[-n_tail:]

    parts: list[torch.Tensor] = []
    for layer_idx in LAYERS:
        layer = hidden_states[layer_idx]                           # (seq_len, hidden_dim)
        parts.append(layer[last_pos])                              # last real-token
        parts.append(layer.index_select(0, tail_idx).mean(0))      # mean of tail

    return torch.cat(parts, dim=0).to(torch.float32).cpu()


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Entry point called by solution.py. The final model does not use
    hand-crafted geometric features.
    """
    if use_geometric:
        raise NotImplementedError("Geometric features are not part of the final model.")
    return aggregate(hidden_states, attention_mask)
