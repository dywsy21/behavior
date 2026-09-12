"""Opt-in intent residual on the FM vector field; the original field is intact."""
from __future__ import annotations

import torch
from torch import nn


class FMIntentAdapter(nn.Module):
    """Cross-attend from frozen, observation-conditioned FM features to intent.

    The output projection starts at zero. Missing/disabled intent has an EXACT
    zero residual even after training. No intent tokens are inserted in the old
    VLM cache; each noise integration step uses the same encoded intent.
    """

    def __init__(self, *, token_dim: int, feature_dim: int, action_dim: int,
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2,
                 max_tokens: int = 256):
        super().__init__()
        if min(token_dim, feature_dim, action_dim, hidden_dim, num_heads, num_layers, max_tokens) < 1:
            raise ValueError("Adapter dimensions must be positive")
        if hidden_dim % num_heads:
            raise ValueError("Adapter hidden_dim must be divisible by num_heads")
        self.max_tokens = int(max_tokens)
        self.token_projection = nn.Linear(token_dim, hidden_dim)
        self.position = nn.Embedding(max_tokens, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, 4 * hidden_dim,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0,
                                                     batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2 * hidden_dim),
                                 nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode(self, embeddings: torch.Tensor, token_mask: torch.Tensor,
               active: torch.Tensor) -> dict[str, torch.Tensor]:
        if embeddings.ndim != 3 or token_mask.shape != embeddings.shape[:2]:
            raise ValueError("Intent embeddings and mask must be [B,T,D] and [B,T]")
        if not 1 <= embeddings.shape[1] <= self.max_tokens:
            raise ValueError("Intent exceeds adapter context; do not silently truncate it")
        if active.shape != embeddings.shape[:1] or not token_mask.bool().any(dim=1).all():
            raise ValueError("Each intent row needs a token and one explicit active flag")
        # Frozen VLM embeddings never receive the FM adapter gradient.
        tokens = self.token_projection(embeddings.detach())
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = tokens + self.position(positions).to(tokens.dtype)
        padding = ~token_mask.bool()
        tokens = self.encoder(tokens, src_key_padding_mask=padding)
        return {"tokens": self.token_norm(tokens), "padding": padding, "active": active.bool()}

    def forward(self, features: torch.Tensor, context: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = context["tokens"].shape[0]
        if features.ndim != 3 or features.shape[0] % batch:
            raise ValueError("FM feature batch must be an integer flow-sample multiple")
        # FMHelper stacks complete B-sized samples N times, NOT repeat_interleave.
        repeats = features.shape[0] // batch
        tokens = context["tokens"].repeat(repeats, 1, 1)
        padding = context["padding"].repeat(repeats, 1)
        active = context["active"].repeat(repeats)
        query = self.feature_projection(features.detach())
        attended, _ = self.cross_attention(self.query_norm(query), tokens, tokens,
                                            key_padding_mask=padding, need_weights=False)
        value = query + attended
        value = value + self.mlp(value)
        residual = self.output(value).float()
        return residual * active[:, None, None].to(residual.dtype)
