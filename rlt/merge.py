"""Gated merge of encoder features with the previous decoder output."""

import torch
from torch import nn

from .layers import RMSNorm


class Merge(nn.Module):
    def __init__(self, dim: int, alpha: float, rms_eps: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.norm_s = RMSNorm(dim, rms_eps)
        self.gate = nn.Linear(2 * dim, dim, bias=True)
        self.state = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self, e_t: torch.Tensor, s_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r_prev = self.norm_s(s_prev)
        gate = torch.sigmoid(self.gate(torch.cat((e_t, r_prev), dim=-1)))
        u_t = e_t + self.alpha * gate * self.state(r_prev)
        return u_t, gate
