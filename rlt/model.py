"""Whole-sequence and incremental recurrent looped transformer."""

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder, EncoderCache
from .layers import RMSNorm
from .memory import Memory, MemoryCache
from .merge import Merge
from .window_cache import WindowCache


@dataclass(frozen=True)
class ModelState:
    """Snapshot after the tokens preceding ``next_position`` are consumed."""

    encoder_cache: EncoderCache
    memory_cache: MemoryCache
    s: torch.Tensor
    decoder_caches: list[WindowCache]
    next_position: int


class RLTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        alpha: float,
        rms_eps: float,
        bos_token_id: int,
        num_memory_groups: int,
    ) -> None:
        super().__init__()
        self.bos_token_id = bos_token_id
        self.encoder = Encoder(
            vocab_size, dim, num_encoder_layers, num_heads, head_dim, rms_eps
        )
        self.memory = Memory(
            dim, num_decoder_layers, num_memory_groups, num_heads, head_dim,
            rms_eps,
        )
        self.merge = Merge(dim, alpha, rms_eps)
        self.decoder = Decoder(
            dim, num_decoder_layers, num_heads, head_dim, window_size, rms_eps
        )
        self.s_star = nn.Parameter(torch.randn(dim))
        self.norm_o = RMSNorm(dim, rms_eps)
        self.readout = nn.Linear(dim, vocab_size, bias=False)

    def initial_state(self, batch_size: int, device: torch.device) -> ModelState:
        return ModelState(
            encoder_cache=self.encoder.empty_cache(batch_size, device),
            memory_cache=self.memory.empty_cache(batch_size, device),
            s=self.s_star.expand(batch_size, -1),
            decoder_caches=self.decoder.empty_caches(batch_size, device),
            next_position=1,
        )

    def _check_bos(self, first_token: torch.Tensor, position: int) -> None:
        if position == 1 and not torch.all(first_token == self.bos_token_id):
            raise ValueError("the first token of every sequence must be BOS")

    def step(
        self, token: torch.Tensor, state: ModelState
    ) -> tuple[torch.Tensor, ModelState]:
        self._check_bos(token, state.next_position)
        e_t, encoder_cache = self.encoder.step(token, state.encoder_cache)
        position = torch.full(
            (token.shape[0],), state.next_position, device=token.device,
            dtype=torch.long,
        )
        memory_cache = self.memory.append(e_t, position, state.memory_cache)
        u_t, _ = self.merge(e_t, state.s)
        s_t, decoder_caches = self.decoder(
            u_t, state.next_position, self.memory, memory_cache,
            state.decoder_caches,
        )
        logits = self.readout(self.norm_o(s_t))
        return logits, ModelState(
            encoder_cache, memory_cache, s_t, decoder_caches,
            state.next_position + 1,
        )

    def prefill(
        self, tokens: torch.Tensor, state: ModelState
    ) -> tuple[torch.Tensor, ModelState]:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape (B, T) with T >= 1")
        self._check_bos(tokens[:, 0], state.next_position)
        features, encoder_cache = self.encoder.prefill(
            tokens, state.encoder_cache
        )
        memory_cache = state.memory_cache
        s = state.s
        decoder_caches = state.decoder_caches
        outputs = []
        for index in range(tokens.shape[1]):
            position = state.next_position + index
            e_t = features[:, index]
            positions = torch.full(
                (tokens.shape[0],), position, device=tokens.device,
                dtype=torch.long,
            )
            memory_cache = self.memory.append(e_t, positions, memory_cache)
            u_t, _ = self.merge(e_t, s)
            s, decoder_caches = self.decoder(
                u_t, position, self.memory, memory_cache, decoder_caches
            )
            outputs.append(self.readout(self.norm_o(s)))

        return torch.stack(outputs, dim=1), ModelState(
            encoder_cache, memory_cache, s, decoder_caches,
            state.next_position + tokens.shape[1],
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.prefill(
            tokens, self.initial_state(tokens.shape[0], tokens.device)
        )
        return logits
