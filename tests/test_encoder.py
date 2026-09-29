"""Contracts for the causal encoder's parallel and cached schedules."""

import pytest
import torch

from encoder import Encoder


VOCAB_SIZE = 31
MODEL_DIM = 8
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 4
BOS = 2


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


def make_encoder(num_layers=NUM_LAYERS):
    return Encoder(
        vocab_size=VOCAB_SIZE,
        dim=MODEL_DIM,
        num_layers=num_layers,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rms_eps=1e-6,
    )


def make_tokens(batch_size=2, length=20):
    tokens = torch.randint(3, VOCAB_SIZE, (batch_size, length))
    tokens[:, 0] = BOS
    return tokens


def positions_for(tokens):
    return torch.arange(1, tokens.shape[1] + 1).expand(tokens.shape[0], -1)


def empty_cache(encoder, tokens):
    return encoder.empty_cache(batch_size=tokens.shape[0], device=tokens.device)


def encode_incrementally(encoder, tokens):
    cache = empty_cache(encoder, tokens)
    outputs = []
    for index in range(tokens.shape[1]):
        output, cache = encoder.step(tokens[:, index], cache)
        outputs.append(output)
    return torch.stack(outputs, dim=1), cache


def assert_caches_close(actual, expected):
    assert actual.next_position == expected.next_position
    assert len(actual.layers) == len(expected.layers) == NUM_LAYERS
    for actual_layer, expected_layer in zip(actual.layers, expected.layers):
        assert actual_layer.key.shape == expected_layer.key.shape
        assert actual_layer.value.shape == expected_layer.value.shape
        torch.testing.assert_close(
            actual_layer.key, expected_layer.key, rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            actual_layer.value, expected_layer.value, rtol=0, atol=1e-12
        )


def test_parallel_matches_every_incremental_position():
    encoder = make_encoder()
    tokens = make_tokens()

    parallel = encoder.forward(tokens, positions_for(tokens))
    incremental, cache = encode_incrementally(encoder, tokens)

    assert parallel.shape == incremental.shape == (2, 20, MODEL_DIM)
    assert parallel.dtype == incremental.dtype == torch.float64
    torch.testing.assert_close(parallel, incremental, rtol=0, atol=1e-12)
    assert cache.next_position == 21


@pytest.mark.parametrize("split", [1, 5, 10, 19])
def test_chunked_prefill_matches_parallel_at_every_position(split):
    encoder = make_encoder()
    tokens = make_tokens()
    parallel = encoder.forward(tokens, positions_for(tokens))

    first, cache = encoder.prefill(tokens[:, :split], empty_cache(encoder, tokens))
    second, cache = encoder.prefill(tokens[:, split:], cache)
    chunked = torch.cat((first, second), dim=1)

    assert first.shape[1] == split
    assert second.shape[1] == 20 - split
    torch.testing.assert_close(parallel, chunked, rtol=0, atol=1e-12)
    assert cache.next_position == 21


def test_incremental_cache_matches_full_parallel_prefill():
    encoder = make_encoder()
    tokens = make_tokens()

    _, parallel_cache = encoder.prefill(tokens, empty_cache(encoder, tokens))
    _, incremental_cache = encode_incrementally(encoder, tokens)

    assert parallel_cache.next_position == incremental_cache.next_position == 21
    for layer in parallel_cache.layers:
        assert layer.key.shape == (2, NUM_HEADS, 20, HEAD_DIM)
        assert layer.value.shape == (2, NUM_HEADS, 20, HEAD_DIM)
    assert_caches_close(incremental_cache, parallel_cache)


def test_changing_token_cannot_change_earlier_encoder_outputs():
    encoder = make_encoder()
    tokens = make_tokens(batch_size=1)
    changed_tokens = tokens.clone()
    changed_tokens[0, 9] = (tokens[0, 9] - 3 + 1) % (VOCAB_SIZE - 3) + 3

    original = encoder.forward(tokens, positions_for(tokens))
    changed = encoder.forward(changed_tokens, positions_for(changed_tokens))

    assert torch.equal(original[:, :9], changed[:, :9])
    assert not torch.equal(original[:, 9], changed[:, 9])


def test_rope_makes_preceding_token_order_visible_to_one_layer():
    encoder = make_encoder(num_layers=1)
    tokens = torch.tensor([[BOS, 3, 4, 5]])
    reordered = torch.tensor([[BOS, 4, 3, 5]])

    original = encoder.forward(tokens, positions_for(tokens))
    changed = encoder.forward(reordered, positions_for(reordered))

    assert torch.equal(original[:, 0], changed[:, 0])
    assert torch.max(torch.abs(original[:, -1] - changed[:, -1])) > 1e-8


def test_step_uses_next_absolute_position_from_cache():
    encoder = make_encoder()
    tokens = make_tokens(batch_size=1, length=8)
    full_outputs = encoder.forward(tokens, positions_for(tokens))
    _, full_cache = encoder.prefill(tokens, empty_cache(encoder, tokens))

    _, prefix_cache = encoder.prefill(tokens[:, :7], empty_cache(encoder, tokens))
    assert prefix_cache.next_position == 8
    stepped, stepped_cache = encoder.step(tokens[:, 7], prefix_cache)

    torch.testing.assert_close(stepped, full_outputs[:, 7], rtol=0, atol=1e-12)
    assert stepped_cache.next_position == 9
    for stepped_layer, full_layer in zip(stepped_cache.layers, full_cache.layers):
        torch.testing.assert_close(
            stepped_layer.key[:, :, 7], full_layer.key[:, :, 7], rtol=0, atol=1e-12
        )


def test_batch_sequences_have_isolated_outputs_and_caches():
    encoder = make_encoder()
    tokens = make_tokens()
    changed_tokens = tokens.clone()
    changed_tokens[0, 7] = (tokens[0, 7] - 3 + 1) % (VOCAB_SIZE - 3) + 3

    original = encoder.forward(tokens, positions_for(tokens))
    changed = encoder.forward(changed_tokens, positions_for(changed_tokens))
    _, original_cache = encoder.prefill(tokens, empty_cache(encoder, tokens))
    _, changed_cache = encoder.prefill(
        changed_tokens, empty_cache(encoder, changed_tokens)
    )

    assert torch.equal(original[1], changed[1])
    assert not torch.equal(original[0, 7], changed[0, 7])
    for original_layer, changed_layer in zip(
        original_cache.layers, changed_cache.layers
    ):
        assert torch.equal(original_layer.key[1], changed_layer.key[1])
        assert torch.equal(original_layer.value[1], changed_layer.value[1])


def test_forward_and_full_parallel_prefill_are_bit_identical():
    encoder = make_encoder()
    tokens = make_tokens()

    forward = encoder.forward(tokens, positions_for(tokens))
    prefill, cache = encoder.prefill(tokens, empty_cache(encoder, tokens))

    assert torch.equal(forward, prefill)
    assert cache.next_position == 21
