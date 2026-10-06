"""Encoder-derived key/value memory for decoder cross-attention."""

from dataclasses import dataclass

import torch
from torch import nn

from .layers import RMSNorm, apply_rope


@dataclass(frozen=True)
class MemoryGroup:
    key: torch.Tensor
    value: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class MemoryCache:
    positions: torch.Tensor
    groups: tuple[MemoryGroup, ...]


class Memory(nn.Module):
    """Project encoder features into one or per-layer memory groups."""

    def __init__(
        self,
        dim: int,
        num_decoder_layers: int,
        num_groups: int,
        num_heads: int,
        head_dim: int,
        rms_eps: float,
    ) -> None:
        super().__init__()
        if num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be positive")
        if num_groups not in (1, num_decoder_layers):
            raise ValueError("num_groups must be 1 or num_decoder_layers")
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

        self.num_decoder_layers = num_decoder_layers
        self.num_groups = num_groups
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm_e = RMSNorm(dim, rms_eps)
        self.keys = nn.ModuleList(
            nn.Linear(dim, num_heads * head_dim, bias=False)
            for _ in range(num_groups)
        )
        self.values = nn.ModuleList(
            nn.Linear(dim, num_heads * head_dim, bias=False)
            for _ in range(num_groups)
        )
        indices = torch.arange(0, head_dim, 2, dtype=torch.get_default_dtype())
        self.register_buffer("inv_freq", 10000.0 ** (-indices / head_dim))

    def _heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = projected.shape
        return projected.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def empty_cache(self, batch_size: int, device: torch.device) -> MemoryCache:
        shape = (batch_size, self.num_heads, 0, self.head_dim)
        dtype = self.norm_e.weight.dtype
        positions = torch.empty((batch_size, 0), device=device, dtype=torch.long)
        groups = tuple(
            MemoryGroup(
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
                positions,
            )
            for _ in range(self.num_groups)
        )
        return MemoryCache(positions, groups)

    def forward(self, e: torch.Tensor, positions: torch.Tensor) -> MemoryCache:
        if e.ndim != 3 or positions.shape != e.shape[:2]:
            raise ValueError("e and positions must have shapes (B, T, d) and (B, T)")

        normalized = self.norm_e(e)
        frequencies = self.inv_freq.to(dtype=normalized.dtype)
        stored_positions = positions.clone()
        groups = tuple(
            MemoryGroup(
                apply_rope(self._heads(key(normalized)), positions, frequencies),
                self._heads(value(normalized)),
                stored_positions,
            )
            for key, value in zip(self.keys, self.values)
        )
        return MemoryCache(stored_positions, groups)

    def append(
        self, e_t: torch.Tensor, position: torch.Tensor, cache: MemoryCache
    ) -> MemoryCache:
        if e_t.ndim != 2 or position.shape != (e_t.shape[0],):
            raise ValueError("e_t and position must have shapes (B, d) and (B,)")
        if cache.positions.shape[0] != e_t.shape[0]:
            raise ValueError("cache and new entry must have the same batch size")

        expected = (
            torch.ones_like(position)
            if cache.positions.shape[1] == 0
            else cache.positions[:, -1] + 1
        )
        if not torch.equal(position, expected):
            raise ValueError("position must be one past the last stored position")

        entry = self.forward(e_t[:, None, :], position[:, None])
        positions = torch.cat((cache.positions, entry.positions), dim=1)
        groups = tuple(
            MemoryGroup(
                torch.cat((old.key, new.key), dim=2),
                torch.cat((old.value, new.value), dim=2),
                positions,
            )
            for old, new in zip(cache.groups, entry.groups)
        )
        return MemoryCache(positions, groups)

    def for_layer(
        self, cache: MemoryCache, layer_index: int, through_position: int
    ) -> MemoryGroup:
        if not 0 <= layer_index < self.num_decoder_layers:
            raise ValueError("layer_index is outside the decoder")
        if not 1 <= through_position <= cache.positions.shape[1]:
            raise ValueError("through_position is outside stored memory")

        group_index = 0 if self.num_groups == 1 else layer_index
        group = cache.groups[group_index]
        return MemoryGroup(
            group.key[:, :, :through_position],
            group.value[:, :, :through_position],
            cache.positions[:, :through_position],
        )
