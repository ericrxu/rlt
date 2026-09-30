"""Contracts for encoder-derived, prefix-restricted key/value memory."""

import pytest
import torch

from memory import Memory


MODEL_DIM = 8
NUM_DECODER_LAYERS = 3
NUM_HEADS = 2
HEAD_DIM = 4
LENGTH = 20


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


def make_memory(num_groups=1):
    return Memory(
        dim=MODEL_DIM,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_groups=num_groups,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rms_eps=1e-6,
    )


def make_features(batch_size=2, length=LENGTH):
    return torch.randn(batch_size, length, MODEL_DIM)


def positions_for(features):
    return torch.arange(1, features.shape[1] + 1).expand(features.shape[0], -1)


def assert_group_equal(actual, expected):
    assert actual.key.shape == expected.key.shape
    assert actual.value.shape == expected.value.shape
    assert actual.positions.shape == expected.positions.shape
    assert torch.equal(actual.positions, expected.positions)
    assert torch.equal(actual.key, expected.key)
    assert torch.equal(actual.value, expected.value)


def test_longer_memory_does_not_change_prefix():
    memory = make_memory()
    features = make_features()
    positions = positions_for(features)
    changed_features = features.clone()
    changed_features[:, 10:] += 10.0

    original = memory.forward(features, positions)
    changed = memory.forward(changed_features, positions)
    first_ten = memory.for_layer(original, 0, 10)
    changed_first_ten = memory.for_layer(changed, 0, 10)
    original_full = memory.for_layer(original, 0, LENGTH)
    changed_full = memory.for_layer(changed, 0, LENGTH)

    assert first_ten.key.shape == (2, NUM_HEADS, 10, HEAD_DIM)
    assert first_ten.value.shape == (2, NUM_HEADS, 10, HEAD_DIM)
    assert first_ten.key.dtype == first_ten.value.dtype == torch.float64
    assert_group_equal(first_ten, changed_first_ten)
    assert not torch.equal(original_full.key[:, :, 10], changed_full.key[:, :, 10])
    assert not torch.equal(original_full.value[:, :, 10], changed_full.value[:, :, 10])


@pytest.mark.parametrize("num_groups", [1, NUM_DECODER_LAYERS])
def test_parallel_memory_matches_one_entry_at_a_time(num_groups):
    memory = make_memory(num_groups)
    features = make_features()
    positions = positions_for(features)
    parallel = memory.forward(features, positions)
    appended = memory.empty_cache(batch_size=features.shape[0], device=features.device)

    for index in range(LENGTH):
        previous = appended
        snapshots = []
        if index:
            for layer in range(NUM_DECODER_LAYERS):
                group = memory.for_layer(previous, layer, index)
                snapshots.append(
                    (group.key.clone(), group.value.clone(), group.positions.clone())
                )
        appended = memory.append(features[:, index], positions[:, index], appended)
        if index:
            for layer in range(NUM_DECODER_LAYERS):
                old_group = memory.for_layer(previous, layer, index)
                new_prefix = memory.for_layer(appended, layer, index)
                key, value, stored_positions = snapshots[layer]
                assert torch.equal(old_group.key, key)
                assert torch.equal(old_group.value, value)
                assert torch.equal(old_group.positions, stored_positions)
                assert_group_equal(new_prefix, old_group)

    for layer in range(NUM_DECODER_LAYERS):
        actual = memory.for_layer(appended, layer, LENGTH)
        expected = memory.for_layer(parallel, layer, LENGTH)
        assert actual.key.shape == expected.key.shape == (2, NUM_HEADS, LENGTH, HEAD_DIM)
        assert actual.value.shape == expected.value.shape == (2, NUM_HEADS, LENGTH, HEAD_DIM)
        assert torch.equal(actual.positions, expected.positions)
        torch.testing.assert_close(actual.key, expected.key, rtol=0, atol=1e-12)
        torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=1e-12)


def test_memory_keys_depend_on_explicit_position():
    memory = make_memory()
    vector = torch.randn(1, 1, MODEL_DIM)
    features = vector.expand(1, 2, -1).clone()
    cache = memory.forward(features, torch.tensor([[3, 7]]))
    keys = memory.for_layer(cache, 0, 2).key

    assert not torch.equal(keys[:, :, 0], keys[:, :, 1])


def test_memory_values_do_not_depend_on_position():
    memory = make_memory()
    vector = torch.randn(1, 1, MODEL_DIM)
    features = vector.expand(1, 2, -1).clone()
    cache = memory.forward(features, torch.tensor([[3, 7]]))
    values = memory.for_layer(cache, 0, 2).value

    assert torch.equal(values[:, :, 0], values[:, :, 1])


def test_key_uses_position_instead_of_tensor_index():
    memory = make_memory()
    vector = torch.randn(MODEL_DIM)
    alone = memory.forward(vector.reshape(1, 1, MODEL_DIM), torch.tensor([[7]]))

    batch = make_features(batch_size=2, length=3)
    batch[1, 2] = vector
    positions = torch.tensor([[5, 6, 7], [5, 6, 7]])
    within_batch = memory.forward(batch, positions)

    alone_key = memory.for_layer(alone, 0, 1).key[0, :, 0]
    batched_key = memory.for_layer(within_batch, 0, 3).key[1, :, 2]
    torch.testing.assert_close(alone_key, batched_key, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "prior_positions, invalid_position",
    [([], 2), ([1], 1), ([1], 3)],
)
def test_append_rejects_nonconsecutive_position(prior_positions, invalid_position):
    memory = make_memory()
    features = make_features(length=2)
    cache = memory.empty_cache(batch_size=features.shape[0], device=features.device)
    for index, position in enumerate(prior_positions):
        positions = torch.full((features.shape[0],), position, dtype=torch.long)
        cache = memory.append(features[:, index], positions, cache)

    invalid_positions = torch.full(
        (features.shape[0],), invalid_position, dtype=torch.long
    )
    with pytest.raises(ValueError):
        memory.append(features[:, len(prior_positions)], invalid_positions, cache)


def test_memory_group_mapping_and_distinct_projections():
    features = make_features(length=5)
    positions = positions_for(features)

    shared_memory = make_memory(num_groups=1)
    shared_cache = shared_memory.forward(features, positions)
    shared = shared_memory.for_layer(shared_cache, 0, 5)
    for layer in range(1, NUM_DECODER_LAYERS):
        assert_group_equal(shared_memory.for_layer(shared_cache, layer, 5), shared)

    separate_memory = make_memory(num_groups=NUM_DECODER_LAYERS)
    separate_cache = separate_memory.forward(features, positions)
    groups = [
        separate_memory.for_layer(separate_cache, layer, 5)
        for layer in range(NUM_DECODER_LAYERS)
    ]
    for left in range(NUM_DECODER_LAYERS):
        for right in range(left + 1, NUM_DECODER_LAYERS):
            assert not torch.equal(groups[left].key, groups[right].key)
            assert not torch.equal(groups[left].value, groups[right].value)


@pytest.mark.parametrize("num_groups", [0, 2, 4])
def test_unsupported_group_count_raises_value_error(num_groups):
    with pytest.raises(ValueError):
        make_memory(num_groups=num_groups)


def test_key_and_value_gradients_reach_encoder_features():
    memory = make_memory(num_groups=NUM_DECODER_LAYERS)
    features = make_features(length=5).requires_grad_()
    cache = memory.forward(features, positions_for(features))

    for layer in range(NUM_DECODER_LAYERS):
        group = memory.for_layer(cache, layer, 5)
        for output in (group.key, group.value):
            gradient = torch.autograd.grad(output.square().sum(), features, retain_graph=True)[0]
            assert gradient is not None
            assert torch.count_nonzero(gradient) > 0


def test_batched_sequences_have_isolated_memory():
    memory = make_memory(num_groups=NUM_DECODER_LAYERS)
    features = make_features(length=5)
    positions = positions_for(features)
    changed_features = features.clone()
    changed_features[0, 2] += 10.0

    original = memory.forward(features, positions)
    changed = memory.forward(changed_features, positions)
    for layer in range(NUM_DECODER_LAYERS):
        original_group = memory.for_layer(original, layer, 5)
        changed_group = memory.for_layer(changed, layer, 5)
        assert torch.equal(original_group.key[1], changed_group.key[1])
        assert torch.equal(original_group.value[1], changed_group.value[1])
        assert not torch.equal(original_group.key[0, :, 2], changed_group.key[0, :, 2])
        assert not torch.equal(original_group.value[0, :, 2], changed_group.value[0, :, 2])


@pytest.mark.parametrize("through_position", [1, 2, 10, LENGTH])
def test_for_layer_returns_exact_prefix(through_position):
    memory = make_memory(num_groups=NUM_DECODER_LAYERS)
    features = make_features()
    cache = memory.forward(features, positions_for(features))

    for layer in range(NUM_DECODER_LAYERS):
        full = memory.for_layer(cache, layer, LENGTH)
        prefix = memory.for_layer(cache, layer, through_position)
        expected_positions = torch.arange(1, through_position + 1).expand(2, -1)

        assert prefix.key.shape == (2, NUM_HEADS, through_position, HEAD_DIM)
        assert prefix.value.shape == (2, NUM_HEADS, through_position, HEAD_DIM)
        assert torch.equal(prefix.positions, expected_positions)
        assert torch.equal(prefix.key, full.key[:, :, :through_position])
        assert torch.equal(prefix.value, full.value[:, :, :through_position])


@pytest.mark.parametrize("through_position", [0, -1, LENGTH + 1])
def test_for_layer_rejects_position_outside_stored_prefix(through_position):
    memory = make_memory()
    features = make_features()
    cache = memory.forward(features, positions_for(features))

    with pytest.raises(ValueError):
        memory.for_layer(cache, 0, through_position)


@pytest.mark.parametrize("layer_index", [-1, NUM_DECODER_LAYERS, NUM_DECODER_LAYERS + 1])
def test_for_layer_rejects_invalid_layer_index(layer_index):
    memory = make_memory()
    features = make_features(length=5)
    cache = memory.forward(features, positions_for(features))

    with pytest.raises(ValueError):
        memory.for_layer(cache, layer_index, 5)
