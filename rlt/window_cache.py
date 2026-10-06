"""One decoder layer's fixed-size sliding-window key/value cache."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WindowView:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True)
class WindowCache:
    key: torch.Tensor
    value: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    next_position: int

    @classmethod
    def empty(
        cls,
        window_size: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "WindowCache":
        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        slots = window_size - 1
        shape = (batch_size, num_heads, slots, head_dim)
        metadata_shape = (batch_size, slots)
        return cls(
            key=torch.zeros(shape, device=device, dtype=dtype),
            value=torch.zeros(shape, device=device, dtype=dtype),
            valid=torch.zeros(metadata_shape, device=device, dtype=torch.bool),
            positions=torch.zeros(metadata_shape, device=device, dtype=torch.long),
            next_position=1,
        )

    def read(
        self, current_key: torch.Tensor, current_value: torch.Tensor, position: int
    ) -> WindowView:
        if position != self.next_position:
            raise ValueError("position must equal next_position")
        expected_shape = self.key.shape[:2] + self.key.shape[3:]
        if current_key.shape != expected_shape or current_value.shape != expected_shape:
            raise ValueError("current key and value must have shape (B, H, D)")

        batch_size = self.key.shape[0]
        current_valid = torch.ones((batch_size, 1), device=self.valid.device, dtype=torch.bool)
        current_position = torch.full(
            (batch_size, 1), position, device=self.positions.device, dtype=torch.long
        )
        return WindowView(
            key=torch.cat((self.key, current_key.unsqueeze(2)), dim=2),
            value=torch.cat((self.value, current_value.unsqueeze(2)), dim=2),
            valid=torch.cat((self.valid, current_valid), dim=1),
            positions=torch.cat((self.positions, current_position), dim=1),
        )

    def update(
        self, current_key: torch.Tensor, current_value: torch.Tensor, position: int
    ) -> "WindowCache":
        view = self.read(current_key, current_value, position)
        return WindowCache(
            key=view.key[:, :, 1:],
            value=view.value[:, :, 1:],
            valid=view.valid[:, 1:],
            positions=view.positions[:, 1:],
            next_position=position + 1,
        )
