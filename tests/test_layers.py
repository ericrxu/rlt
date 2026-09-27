"""Contracts for the primitive RLT layers."""

import pytest
import torch

from layers import Attention, FFN, RMSNorm, apply_rope


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


def test_rmsnorm_unit_rms_with_unit_scale():
    norm = RMSNorm(dim=4, eps=0.0)
    parameters = list(norm.parameters())
    assert len(parameters) == 1
    with torch.no_grad():
        parameters[0].fill_(1.0)

    x = torch.randn(2, 3, 4) + 2.0
    output = norm(x)
    rms = output.square().mean(dim=-1).sqrt()

    assert output.dtype == torch.float64
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=0, atol=1e-12)


def test_rmsnorm_has_no_bias_and_does_not_center():
    norm = RMSNorm(dim=4, eps=0.0)
    parameters = dict(norm.named_parameters())
    assert len(parameters) == 1
    assert all("bias" not in name for name in parameters)
    with torch.no_grad():
        next(iter(parameters.values())).fill_(1.0)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    expected = x / x.square().mean(dim=-1, keepdim=True).sqrt()
    output = norm(x)

    torch.testing.assert_close(output, expected, rtol=0, atol=1e-12)
    assert output.mean() > 0


def test_ffn_preserves_shape_and_is_position_wise():
    ffn = FFN(dim=4, hidden_dim=9, activation=torch.nn.GELU(), bias=False)
    x = torch.randn(2, 5, 4)
    original = ffn(x)
    changed_x = x.clone()
    changed_x[:, 3, :] += 10.0
    changed = ffn(changed_x)

    assert original.shape == x.shape
    assert original.dtype == torch.float64
    assert torch.equal(original[:, :3, :], changed[:, :3, :])
    assert torch.equal(original[:, 4:, :], changed[:, 4:, :])
    assert not torch.equal(original[:, 3, :], changed[:, 3, :])


def test_rope_uses_absolute_position_not_tensor_index():
    vector = torch.randn(4)
    inv_freq = torch.tensor([0.2, 0.05])
    alone = apply_rope(vector.reshape(1, 1, 1, 4),
                       torch.tensor([[7]]), inv_freq)

    batch = torch.randn(2, 1, 3, 4)
    batch[1, 0, 2] = vector
    positions = torch.tensor([[11, 12, 13], [20, 21, 7]])
    within_batch = apply_rope(batch, positions, inv_freq)

    assert torch.equal(alone[0, 0, 0], within_batch[1, 0, 2])


def test_rope_query_key_score_depends_on_relative_distance():
    query = torch.tensor([[[[1.0, 0.0, 0.5, -0.3]]]])
    key = torch.tensor([[[[0.0, 1.0, -0.2, 0.7]]]])
    inv_freq = torch.tensor([0.2, 0.05])

    def score(query_position, key_position):
        rotated_query = apply_rope(query, torch.tensor([[query_position]]), inv_freq)
        rotated_key = apply_rope(key, torch.tensor([[key_position]]), inv_freq)
        return (rotated_query * rotated_key).sum()

    first = score(2, 5)
    shifted = score(9, 12)
    different_distance = score(2, 6)

    torch.testing.assert_close(first, shifted, rtol=0, atol=1e-12)
    assert not torch.isclose(first, different_distance, rtol=0, atol=1e-6)


def test_rope_preserves_vector_norm():
    x = torch.randn(2, 3, 4, 6)
    positions = torch.tensor([[1, 2, 7, 11], [3, 8, 13, 21]])
    inv_freq = torch.tensor([0.2, 0.05, 0.01])

    rotated = apply_rope(x, positions, inv_freq)

    torch.testing.assert_close(
        rotated.linalg.vector_norm(dim=-1), x.linalg.vector_norm(dim=-1),
        rtol=0, atol=1e-12,
    )


def test_rope_rejects_odd_head_dim():
    x = torch.randn(1, 1, 2, 5)
    with pytest.raises(ValueError):
        apply_rope(x, torch.tensor([[1, 2]]), torch.tensor([0.2, 0.05]))


def test_attention_full_causal_sequence_matches_growing_prefixes():
    attention = Attention(num_heads=2, head_dim=4, model_dim=7, output_bias=False)
    q = torch.randn(2, 2, 5, 4)
    k = torch.randn(2, 2, 5, 4)
    v = torch.randn(2, 2, 5, 4)
    causal_mask = torch.ones(5, 5, dtype=torch.bool).tril()[None, None]

    full = attention(q, k, v, causal_mask)
    incremental = torch.cat([
        attention(
            q[:, :, step:step + 1],
            k[:, :, :step + 1],
            v[:, :, :step + 1],
            torch.ones(1, 1, 1, step + 1, dtype=torch.bool),
        )
        for step in range(5)
    ], dim=1)

    assert full.shape == (2, 5, 7)
    torch.testing.assert_close(full, incremental, rtol=0, atol=1e-12)


def test_future_key_value_change_cannot_affect_earlier_queries():
    attention = Attention(num_heads=2, head_dim=4, model_dim=7, output_bias=False)
    q = torch.randn(2, 2, 6, 4)
    k = torch.randn(2, 2, 6, 4)
    v = torch.randn(2, 2, 6, 4)
    causal_mask = torch.ones(6, 6, dtype=torch.bool).tril()[None, None]
    original = attention(q, k, v, causal_mask)

    changed_k = k.clone()
    changed_v = v.clone()
    changed_k[:, :, 3] += 100.0
    changed_v[:, :, 3] += 100.0
    changed = attention(q, changed_k, changed_v, causal_mask)

    assert torch.equal(original[:, :3], changed[:, :3])
    assert not torch.equal(original[:, 3:], changed[:, 3:])


def test_fully_masked_key_never_affects_attention_output():
    attention = Attention(num_heads=2, head_dim=4, model_dim=7, output_bias=False)
    q = torch.randn(2, 2, 3, 4)
    k = torch.randn(2, 2, 5, 4)
    v = torch.randn(2, 2, 5, 4)
    mask = torch.ones(2, 1, 3, 5, dtype=torch.bool)
    mask[:, :, :, 2] = False
    original = attention(q, k, v, mask)

    changed_k = k.clone()
    changed_v = v.clone()
    changed_k[:, :, 2] += 100.0
    changed_v[:, :, 2] += 100.0
    changed = attention(q, changed_k, changed_v, mask)

    assert torch.equal(original, changed)


def test_fully_masked_query_raises_value_error():
    attention = Attention(num_heads=2, head_dim=4, model_dim=7, output_bias=False)
    q = torch.randn(2, 2, 3, 4)
    k = torch.randn(2, 2, 4, 4)
    v = torch.randn(2, 2, 4, 4)
    mask = torch.ones(2, 1, 3, 4, dtype=torch.bool)
    mask[1, :, 1, :] = False

    with pytest.raises(ValueError):
        attention(q, k, v, mask)
