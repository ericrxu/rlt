"""Contracts for one-token decoder layers and their sequential layer stack."""

import pytest
import torch

from decoder import Decoder, DecoderLayer
from layers import apply_rope
from memory import Memory


DIM = 8
HEADS = 2
HEAD_DIM = 4
EPS = 1e-6


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


def make_decoder(num_layers=2, window_size=3):
    return Decoder(
        dim=DIM,
        num_layers=num_layers,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        window_size=window_size,
        rms_eps=EPS,
    )


def make_memory(num_layers=2, num_groups=1, batch_size=2, length=20):
    memory = Memory(
        dim=DIM,
        num_decoder_layers=num_layers,
        num_groups=num_groups,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        rms_eps=EPS,
    )
    features = torch.randn(batch_size, length, DIM)
    positions = torch.arange(1, length + 1).expand(batch_size, -1)
    return memory, memory(features, positions)


def heads(projected):
    batch_size = projected.shape[0]
    return projected.reshape(batch_size, 1, HEADS, HEAD_DIM).transpose(1, 2)


def assert_cache_equal(actual, expected):
    assert actual.next_position == expected.next_position
    assert torch.equal(actual.key, expected.key)
    assert torch.equal(actual.value, expected.value)
    assert torch.equal(actual.valid, expected.valid)
    assert torch.equal(actual.positions, expected.positions)


def test_layer_matches_equations_in_swa_cross_attention_ffn_order():
    layer = DecoderLayer(DIM, HEADS, HEAD_DIM, EPS)
    decoder = make_decoder(num_layers=1)
    cache = decoder.empty_caches(2, torch.device("cpu"))[0]
    memory, memory_cache = make_memory(num_layers=1)

    for position in (1, 2):
        _, cache = layer(
            torch.randn(2, DIM), position,
            memory.for_layer(memory_cache, 0, position), cache,
        )

    z = torch.randn(2, DIM)
    position = 3
    group = memory.for_layer(memory_cache, 0, position)
    normalized = layer.norm_s(z)
    frequencies = layer.inv_freq.to(dtype=z.dtype)
    position_tensor = torch.full((2, 1), position, dtype=torch.long)
    q = apply_rope(heads(layer.query(normalized)), position_tensor, frequencies)
    k = apply_rope(heads(layer.key(normalized)), position_tensor, frequencies)
    v = heads(layer.value(normalized))
    view = cache.read(k.squeeze(2), v.squeeze(2), position)
    b = z + layer.attention(q, view.key, view.value, view.valid[:, None, None, :]).squeeze(1)
    memory_q = apply_rope(
        heads(layer.memory_query(layer.norm_m(b))), position_tensor, frequencies
    )
    memory_mask = group.positions[:, None, None, :] <= position
    a = b + layer.memory_attention(
        memory_q, group.key, group.value, memory_mask
    ).squeeze(1)
    expected = a + layer.ffn(layer.norm_d(a))

    actual, actual_cache = layer(z, position, group, cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert_cache_equal(actual_cache, cache.update(k.squeeze(2), v.squeeze(2), position))


def test_each_layer_retains_key_and_value_from_its_input():
    decoder = make_decoder(num_layers=2, window_size=3)
    memory, memory_cache = make_memory(num_layers=2, num_groups=2)
    caches = decoder.empty_caches(2, torch.device("cpu"))
    _, caches = decoder(
        torch.randn(2, DIM), 1, memory, memory_cache, caches
    )

    z = torch.randn(2, DIM)
    u = z
    position_tensor = torch.full((2, 1), 2, dtype=torch.long)
    manually_updated = []
    for layer_index, layer in enumerate(decoder.layers):
        layer_input = z
        normalized = layer.norm_s(layer_input)
        expected_key = apply_rope(
            heads(layer.key(normalized)),
            position_tensor,
            layer.inv_freq.to(dtype=normalized.dtype),
        ).squeeze(2)
        expected_value = heads(layer.value(normalized)).squeeze(2)

        z, updated = layer(
            layer_input, 2, memory.for_layer(memory_cache, layer_index, 2),
            caches[layer_index],
        )
        assert torch.equal(updated.key[:, :, -1], expected_key)
        assert torch.equal(updated.value[:, :, -1], expected_value)
        manually_updated.append(updated)

    _, returned_caches = decoder(u, 2, memory, memory_cache, caches)
    for returned, expected in zip(returned_caches, manually_updated):
        assert_cache_equal(returned, expected)


def test_decoder_chains_layers_with_their_own_memory_groups():
    decoder = make_decoder(num_layers=3)
    memory, memory_cache = make_memory(num_layers=3, num_groups=3)
    caches = decoder.empty_caches(2, torch.device("cpu"))

    for position in range(1, 5):
        u = torch.randn(2, DIM)
        z = u
        expected_caches = []
        for layer_index, layer in enumerate(decoder.layers):
            z, updated = layer(
                z, position,
                memory.for_layer(memory_cache, layer_index, position),
                caches[layer_index],
            )
            expected_caches.append(updated)

        actual, actual_caches = decoder(u, position, memory, memory_cache, caches)
        assert torch.equal(actual, z)
        assert len(actual_caches) == len(expected_caches) == 3
        for actual_cache, expected_cache in zip(actual_caches, expected_caches):
            assert_cache_equal(actual_cache, expected_cache)
        caches = actual_caches


def test_cross_attention_reads_only_memory_through_current_position():
    decoder = make_decoder(num_layers=2)
    memory = Memory(DIM, 2, 2, HEADS, HEAD_DIM, EPS)
    features = torch.randn(1, 20, DIM)
    positions = torch.arange(1, 21).reshape(1, 20)
    changed_future = features.clone()
    changed_future[:, 5:] += 7.0
    changed_current = features.clone()
    changed_current[:, 4] += 7.0
    original = memory(features, positions)
    future = memory(changed_future, positions)
    current = memory(changed_current, positions)
    caches = decoder.empty_caches(1, torch.device("cpu"))
    u = torch.randn(1, DIM)

    # The cache timeline must reach position 5 before comparing memory variants.
    for position in range(1, 5):
        _, caches = decoder(torch.randn(1, DIM), position, memory, original, caches)

    baseline, _ = decoder(u, 5, memory, original, caches)
    future_output, _ = decoder(u, 5, memory, future, caches)
    current_output, _ = decoder(u, 5, memory, current, caches)
    assert torch.equal(baseline, future_output)
    assert not torch.equal(baseline, current_output)


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("window_size", [1, 3])
def test_swa_reach_is_bounded_without_merge(num_layers, window_size):
    decoder = make_decoder(num_layers=num_layers, window_size=window_size)
    memory, memory_cache = make_memory(num_layers=num_layers, batch_size=1)
    inputs = torch.randn(12, 1, DIM)
    changed_inputs = inputs.clone()
    changed_position = 4
    changed_inputs[changed_position - 1] += 10.0

    def run(sequence):
        caches = decoder.empty_caches(1, torch.device("cpu"))
        outputs = []
        for position, u in enumerate(sequence, start=1):
            output, caches = decoder(u, position, memory, memory_cache, caches)
            outputs.append(output)
        return torch.stack(outputs)

    original = run(inputs)
    changed = run(changed_inputs)
    assert torch.equal(original[: changed_position - 1], changed[: changed_position - 1])
    assert not torch.equal(original[changed_position - 1], changed[changed_position - 1])
    last_reachable = changed_position + num_layers * (window_size - 1)
    assert torch.equal(original[last_reachable:], changed[last_reachable:])


def test_caches_advance_once_per_layer_without_modifying_inputs():
    decoder = make_decoder(num_layers=3, window_size=3)
    memory, memory_cache = make_memory(num_layers=3)
    caches = decoder.empty_caches(2, torch.device("cpu"))

    for position in range(1, 6):
        snapshots = [
            (cache.key.clone(), cache.value.clone(), cache.valid.clone(),
             cache.positions.clone(), cache.next_position)
            for cache in caches
        ]
        _, updated = decoder(
            torch.randn(2, DIM), position, memory, memory_cache, caches
        )
        assert len(updated) == 3
        for old, new, snapshot in zip(caches, updated, snapshots):
            assert new is not old
            assert new.next_position == position + 1
            assert new.valid.shape == new.positions.shape == (2, 2)
            retained = list(range(max(1, position - 1), position + 1))
            for batch_index in range(2):
                assert int(new.valid[batch_index].sum()) == min(position, 2)
                assert new.positions[batch_index, new.valid[batch_index]].tolist() == retained
            assert torch.equal(old.key, snapshot[0])
            assert torch.equal(old.value, snapshot[1])
            assert torch.equal(old.valid, snapshot[2])
            assert torch.equal(old.positions, snapshot[3])
            assert old.next_position == snapshot[4] == position
        caches = updated


def test_gradients_reach_input_memory_and_prior_input_in_window():
    decoder = make_decoder(num_layers=2, window_size=3)
    memory = Memory(DIM, 2, 1, HEADS, HEAD_DIM, EPS)
    features = torch.randn(1, 3, DIM, requires_grad=True)
    positions = torch.arange(1, 4).reshape(1, 3)
    memory_cache = memory(features, positions)
    earlier = torch.randn(1, DIM, requires_grad=True)
    current = torch.randn(1, DIM, requires_grad=True)
    caches = decoder.empty_caches(1, torch.device("cpu"))
    _, caches = decoder(torch.randn(1, DIM), 1, memory, memory_cache, caches)
    _, caches = decoder(earlier, 2, memory, memory_cache, caches)
    output, _ = decoder(current, 3, memory, memory_cache, caches)
    group = memory_cache.groups[0]

    gradients = torch.autograd.grad(
        (output * torch.randn_like(output)).sum(),
        (current, earlier, group.key, group.value),
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0


def test_repeated_or_skipped_position_raises_value_error():
    decoder = make_decoder()
    memory, memory_cache = make_memory()
    caches = decoder.empty_caches(2, torch.device("cpu"))
    u = torch.randn(2, DIM)
    with pytest.raises(ValueError):
        decoder(u, 2, memory, memory_cache, caches)
    _, caches = decoder(u, 1, memory, memory_cache, caches)
    for wrong_position in (1, 3):
        with pytest.raises(ValueError):
            decoder(u, wrong_position, memory, memory_cache, caches)


def test_batch_sequences_have_isolated_outputs_and_caches():
    decoder = make_decoder()
    memory, memory_cache = make_memory()
    inputs = torch.randn(4, 2, DIM)
    changed_inputs = inputs.clone()
    changed_inputs[1, 0] += 10.0

    def run(sequence):
        caches = decoder.empty_caches(2, torch.device("cpu"))
        outputs = []
        for position, u in enumerate(sequence, start=1):
            output, caches = decoder(u, position, memory, memory_cache, caches)
            outputs.append(output)
        return torch.stack(outputs), caches

    original, original_caches = run(inputs)
    changed, changed_caches = run(changed_inputs)
    assert torch.equal(original[:, 1], changed[:, 1])
    assert not torch.equal(original[1, 0], changed[1, 0])
    for old, new in zip(original_caches, changed_caches):
        assert torch.equal(old.key[1], new.key[1])
        assert torch.equal(old.value[1], new.value[1])
        assert torch.equal(old.valid[1], new.valid[1])
        assert torch.equal(old.positions[1], new.positions[1])
