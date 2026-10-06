"""Primitive normalization, feed-forward, position, and attention layers."""

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize by root mean square without centering or a bias."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.weight


class FFN(nn.Module):
    """Apply the supplied activation between two position-wise projections."""

    def __init__(self, dim: int, hidden_dim: int, activation: nn.Module, bias: bool):
        super().__init__()
        self.input = nn.Linear(dim, hidden_dim, bias=bias)
        self.activation = activation
        self.output = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.input(x)))


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    """Rotate adjacent coordinate pairs at explicit absolute positions.

    x has shape (batch, heads, tokens, head_dim); positions has shape
    (batch, tokens), and inv_freq contains one frequency per coordinate pair.
    """
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    if inv_freq.numel() != x.shape[-1] // 2:
        raise ValueError("RoPE needs one frequency per coordinate pair")

    angles = positions[:, None, :, None] * inv_freq
    cos = angles.cos()
    sin = angles.sin()
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    first, second = pairs.unbind(dim=-1)
    rotated = torch.stack((first * cos - second * sin,
                           first * sin + second * cos), dim=-1)
    return rotated.flatten(start_dim=-2)


class Attention(nn.Module):
    """Scaled dot-product attention with an explicit allowed-key mask."""

    def __init__(
        self, num_heads: int, head_dim: int, model_dim: int, output_bias: bool
    ):
        super().__init__()
        self.head_dim = head_dim
        self.output = nn.Linear(num_heads * head_dim, model_dim,
                                bias=output_bias)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = torch.broadcast_to(mask, scores.shape)
        if not torch.all(allowed.any(dim=-1)):
            raise ValueError("every attention query needs an allowed key")
        weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return self.output(context)
