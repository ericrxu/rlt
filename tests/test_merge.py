"""Contracts for the recurrent state's gated merge with encoder features."""

import pytest
import torch
from torch.nn import functional as F

from layers import RMSNorm
from merge import Merge


MODEL_DIM = 8


@pytest.fixture(autouse=True)
def deterministic_float64():
    previous_dtype = torch.get_default_dtype()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1729)
    yield
    torch.set_default_dtype(previous_dtype)
    torch.use_deterministic_algorithms(previous_determinism)


def make_merge(alpha=0.75, rms_eps=1e-6):
    return Merge(dim=MODEL_DIM, alpha=alpha, rms_eps=rms_eps)


def make_inputs(batch_size=3):
    return torch.randn(batch_size, MODEL_DIM), torch.randn(batch_size, MODEL_DIM)


def test_zero_alpha_returns_encoder_features_bit_identically():
    merge = make_merge(alpha=0.0)
    e_t, s_prev = make_inputs()

    u_t, gate = merge(e_t, s_prev)

    assert u_t.shape == gate.shape == e_t.shape
    assert u_t.dtype == gate.dtype == torch.float64
    assert torch.equal(u_t, e_t)


def test_merge_matches_equations_with_encoder_features_first_in_gate_input():
    merge = make_merge(alpha=0.75)
    e_t, s_prev = make_inputs()

    u_t, gate = merge(e_t, s_prev)
    assert isinstance(merge.norm_s, RMSNorm)
    rms = (s_prev.square().mean(dim=-1, keepdim=True) + merge.norm_s.eps).sqrt()
    r_prev = s_prev / rms * merge.norm_s.weight
    expected_gate = torch.sigmoid(
        F.linear(torch.cat((e_t, r_prev), dim=-1), merge.gate.weight, merge.gate.bias)
    )
    expected_u = e_t + merge.alpha * expected_gate * F.linear(
        r_prev, merge.state.weight
    )

    torch.testing.assert_close(gate, expected_gate, rtol=0, atol=1e-12)
    torch.testing.assert_close(u_t, expected_u, rtol=0, atol=1e-12)


def test_gate_is_strictly_between_zero_and_one():
    merge = make_merge()
    e_t, s_prev = make_inputs(batch_size=32)

    _, gate = merge(e_t, s_prev)

    assert torch.all(gate > 0)
    assert torch.all(gate < 1)


def test_state_gradient_depends_on_feedback_scale():
    e_t, s_prev = make_inputs()
    e_t.requires_grad_()
    s_with_feedback = s_prev.clone().requires_grad_()
    s_without_feedback = s_prev.clone().requires_grad_()

    u_with_feedback, _ = make_merge(alpha=0.75)(e_t, s_with_feedback)
    gradient_with_feedback = torch.autograd.grad(
        u_with_feedback.square().sum(), s_with_feedback
    )[0]
    assert torch.count_nonzero(gradient_with_feedback) > 0

    u_without_feedback, _ = make_merge(alpha=0.0)(e_t, s_without_feedback)
    gradient_without_feedback = torch.autograd.grad(
        u_without_feedback.sum(), s_without_feedback, allow_unused=True
    )[0]
    if gradient_without_feedback is None:
        gradient_without_feedback = torch.zeros_like(s_without_feedback)
    assert torch.equal(gradient_without_feedback, torch.zeros_like(s_without_feedback))


def test_positive_state_scaling_does_not_change_merge_when_eps_is_zero():
    merge = make_merge(rms_eps=0.0)
    e_t, s_prev = make_inputs()

    u_t, gate = merge(e_t, s_prev)
    scaled_u, scaled_gate = merge(e_t, 3.7 * s_prev)

    torch.testing.assert_close(scaled_u, u_t, rtol=0, atol=1e-12)
    torch.testing.assert_close(scaled_gate, gate, rtol=0, atol=1e-12)


def test_merge_parameter_layout_and_zero_gate_bias():
    merge = make_merge()

    assert isinstance(merge.norm_s, RMSNorm)
    assert merge.gate.weight.shape == (MODEL_DIM, 2 * MODEL_DIM)
    assert merge.gate.bias.shape == (MODEL_DIM,)
    assert merge.state.weight.shape == (MODEL_DIM, MODEL_DIM)
    assert merge.state.bias is None
    assert torch.equal(merge.gate.bias, torch.zeros_like(merge.gate.bias))


def test_alpha_is_required_fixed_and_stored():
    with pytest.raises(TypeError):
        Merge(dim=MODEL_DIM, rms_eps=1e-6)

    merge = make_merge(alpha=0.375)
    assert isinstance(merge.alpha, float)
    assert merge.alpha == 0.375
    assert "alpha" not in dict(merge.named_parameters())


def test_batch_sequences_are_isolated():
    merge = make_merge()
    e_t, s_prev = make_inputs(batch_size=2)
    changed_e = e_t.clone()
    changed_s = s_prev.clone()
    changed_e[0] += 10.0
    changed_s[0] -= 10.0

    original_u, original_gate = merge(e_t, s_prev)
    changed_u, changed_gate = merge(changed_e, changed_s)

    assert torch.equal(original_u[1], changed_u[1])
    assert torch.equal(original_gate[1], changed_gate[1])
    assert not torch.equal(original_u[0], changed_u[0])
    assert not torch.equal(original_gate[0], changed_gate[0])
