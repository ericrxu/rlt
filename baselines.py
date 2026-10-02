"""Parameter-matched causal Transformer and GRU state-tracking baselines."""

import torch
from torch import nn

from encoder import Encoder
from layers import RMSNorm


class TransformerBaseline(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, num_encoder_layers: int,
        num_heads: int, head_dim: int, rms_eps: float,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size, dim, num_encoder_layers, num_heads, head_dim, rms_eps
        )
        self.norm = RMSNorm(dim, rms_eps)
        self.readout = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            1, tokens.shape[1] + 1, device=tokens.device
        ).expand(tokens.shape[0], -1)
        return self.readout(self.norm(self.encoder(tokens, positions)))


class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.gru = nn.GRU(dim, dim, num_layers, batch_first=True)
        self.readout = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.embedding(tokens))
        return self.readout(states)
