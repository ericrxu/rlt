"""One-token recurrent decoder layers and their sequential layer stack."""

import torch
from torch import nn

from .layers import Attention, FFN, RMSNorm, apply_rope
from .memory import Memory, MemoryCache, MemoryGroup
from .window_cache import WindowCache


class DecoderLayer(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, head_dim: int, rms_eps: float
    ) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm_s = RMSNorm(dim, rms_eps)
        self.query = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.key = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.value = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.attention = Attention(num_heads, head_dim, dim, output_bias=False)

        self.norm_m = RMSNorm(dim, rms_eps)
        self.memory_query = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.memory_attention = Attention(
            num_heads, head_dim, dim, output_bias=False
        )

        self.norm_d = RMSNorm(dim, rms_eps)
        self.ffn = FFN(dim, 4 * dim, nn.GELU(), bias=False)

        indices = torch.arange(0, head_dim, 2, dtype=torch.get_default_dtype())
        self.register_buffer("inv_freq", 10000.0 ** (-indices / head_dim))

    def _heads(self, projected: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = projected.shape
        return projected.reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        z: torch.Tensor,
        position: int,
        memory: MemoryGroup,
        cache: WindowCache,
    ) -> tuple[torch.Tensor, WindowCache]:
        if z.ndim != 2:
            raise ValueError("layer input must have shape (B, d)")
        positions = torch.full(
            (z.shape[0], 1), position, device=z.device, dtype=torch.long
        )
        normalized = self.norm_s(z)[:, None, :]
        frequencies = self.inv_freq.to(dtype=normalized.dtype)
        q = apply_rope(self._heads(self.query(normalized)), positions, frequencies)
        k = apply_rope(self._heads(self.key(normalized)), positions, frequencies)
        v = self._heads(self.value(normalized))

        current_key = k.squeeze(2)
        current_value = v.squeeze(2)
        window = cache.read(current_key, current_value, position)
        b = z + self.attention(
            q, window.key, window.value, window.valid[:, None, None, :]
        ).squeeze(1)

        memory_q = apply_rope(
            self._heads(self.memory_query(self.norm_m(b)[:, None, :])),
            positions,
            frequencies,
        )
        memory_mask = memory.positions[:, None, None, :] <= position
        a = b + self.memory_attention(
            memory_q, memory.key, memory.value, memory_mask
        ).squeeze(1)
        output = a + self.ffn(self.norm_d(a))

        return output, cache.update(current_key, current_value, position)


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        rms_eps: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.layers = nn.ModuleList(
            DecoderLayer(dim, num_heads, head_dim, rms_eps)
            for _ in range(num_layers)
        )

    def empty_caches(
        self, batch_size: int, device: torch.device
    ) -> list[WindowCache]:
        dtype = self.layers[0].norm_s.weight.dtype
        return [
            WindowCache.empty(
                self.window_size, batch_size, self.num_heads, self.head_dim,
                device, dtype,
            )
            for _ in self.layers
        ]

    def forward(
        self,
        u: torch.Tensor,
        position: int,
        memory: Memory,
        memory_cache: MemoryCache,
        caches: list[WindowCache],
    ) -> tuple[torch.Tensor, list[WindowCache]]:
        if len(caches) != len(self.layers):
            raise ValueError("one window cache is required per decoder layer")

        z = u
        updated_caches = []
        for layer_index, (layer, cache) in enumerate(zip(self.layers, caches)):
            group = memory.for_layer(memory_cache, layer_index, position)
            z, updated = layer(z, position, group, cache)
            updated_caches.append(updated)
        return z, updated_caches
