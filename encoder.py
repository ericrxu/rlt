"""Causal encoder with parallel, chunked, and one-token execution."""

from dataclasses import dataclass

import torch
from torch import nn

from layers import Attention, FFN, RMSNorm, apply_rope


@dataclass(frozen=True)
class EncoderLayerCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class EncoderCache:
    next_position: int
    layers: tuple[EncoderLayerCache, ...]


class EncoderBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, head_dim: int, rms_eps: float
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm_s = RMSNorm(dim, rms_eps)
        self.query = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.key = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.value = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.attention = Attention(num_heads, head_dim, dim, output_bias=False)
        self.norm_f = RMSNorm(dim, rms_eps)
        self.ffn = FFN(dim, 4 * dim, nn.GELU(), bias=False)
        indices = torch.arange(0, head_dim, 2, dtype=torch.get_default_dtype())
        self.register_buffer("inv_freq", 10000.0 ** (-indices / head_dim))

    def _heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = projected.shape
        return projected.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, z: torch.Tensor, positions: torch.Tensor, cache: EncoderLayerCache
    ) -> tuple[torch.Tensor, EncoderLayerCache]:
        n = self.norm_s(z)
        frequencies = self.inv_freq.to(dtype=n.dtype)
        q = apply_rope(self._heads(self.query(n)), positions, frequencies)
        current_key = apply_rope(self._heads(self.key(n)), positions, frequencies)
        current_value = self._heads(self.value(n))

        key = torch.cat((cache.key, current_key), dim=2)
        value = torch.cat((cache.value, current_value), dim=2)
        previous = cache.key.shape[2]
        current = z.shape[1]
        query_indices = previous + torch.arange(current, device=z.device)
        key_indices = torch.arange(previous + current, device=z.device)
        causal_mask = (key_indices[None, :] <= query_indices[:, None])[None, None]

        b = z + self.attention(q, key, value, causal_mask)
        output = b + self.ffn(self.norm_f(b))
        return output, EncoderLayerCache(key, value)


class Encoder(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, num_layers: int, num_heads: int,
        head_dim: int, rms_eps: float,
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            EncoderBlock(dim, num_heads, head_dim, rms_eps)
            for _ in range(num_layers)
        )

    def empty_cache(self, batch_size: int, device: torch.device) -> EncoderCache:
        shape = (batch_size, self.num_heads, 0, self.head_dim)
        dtype = self.embedding.weight.dtype
        layers = tuple(
            EncoderLayerCache(
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            for _ in self.layers
        )
        return EncoderCache(next_position=1, layers=layers)

    def _encode(
        self, tokens: torch.Tensor, positions: torch.Tensor, cache: EncoderCache
    ) -> tuple[torch.Tensor, EncoderCache]:
        z = self.embedding(tokens)
        updated_layers = []
        for layer, layer_cache in zip(self.layers, cache.layers):
            z, updated_cache = layer(z, positions, layer_cache)
            updated_layers.append(updated_cache)
        return z, EncoderCache(
            next_position=cache.next_position + tokens.shape[1],
            layers=tuple(updated_layers),
        )

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cache = self.empty_cache(tokens.shape[0], tokens.device)
        output, _ = self._encode(tokens, positions, cache)
        return output

    def prefill(
        self, tokens: torch.Tensor, cache: EncoderCache
    ) -> tuple[torch.Tensor, EncoderCache]:
        positions = (
            cache.next_position
            + torch.arange(tokens.shape[1], device=tokens.device)
        ).expand(tokens.shape[0], -1)
        return self._encode(tokens, positions, cache)

    def step(
        self, token: torch.Tensor, cache: EncoderCache
    ) -> tuple[torch.Tensor, EncoderCache]:
        output, updated_cache = self.prefill(token[:, None], cache)
        return output[:, 0], updated_cache
