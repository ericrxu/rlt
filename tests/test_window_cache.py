"""Contracts for one decoder layer's fixed-size sliding-window K/V cache."""

import pytest
import torch

from window_cache import WindowCache


BATCH_SIZE = 2
NUM_HEADS = 2
HEAD_DIM = 4
LAST_POSITION = 12
WINDOW_SIZES = (1, 2, 3, 5)


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


def empty_cache(window_size):
    return WindowCache.empty(
        window_size=window_size,
        batch_size=BATCH_SIZE,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def make_entry():
    shape = (BATCH_SIZE, NUM_HEADS, HEAD_DIM)
    return torch.randn(shape), torch.randn(shape)


def assert_view_equal(actual, expected):
    assert torch.equal(actual.key, expected.key)
    assert torch.equal(actual.value, expected.value)
    assert torch.equal(actual.valid, expected.valid)
    assert torch.equal(actual.positions, expected.positions)


def assert_empty_slots_have_zero_position(view):
    assert view.valid.dtype == torch.bool
    assert torch.all(view.positions[~view.valid] == 0)


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_read_and_update_follow_exact_window_boundaries(window_size):
    cache = empty_cache(window_size)
    inserted = {}

    assert cache.key.shape == cache.value.shape == (
        BATCH_SIZE, NUM_HEADS, window_size - 1, HEAD_DIM
    )
    assert cache.valid.shape == cache.positions.shape == (BATCH_SIZE, window_size - 1)
    assert cache.next_position == 1
    assert not torch.any(cache.valid)
    assert_empty_slots_have_zero_position(cache)

    for position in range(1, LAST_POSITION + 1):
        current_key, current_value = make_entry()
        inserted[position] = (current_key, current_value)
        view = cache.read(current_key, current_value, position=position)

        assert view.key.shape == view.value.shape == (
            BATCH_SIZE, NUM_HEADS, window_size, HEAD_DIM
        )
        assert view.valid.shape == view.positions.shape == (BATCH_SIZE, window_size)
        assert view.key.dtype == view.value.dtype == torch.float64
        assert torch.equal(view.key[:, :, -1], current_key)
        assert torch.equal(view.value[:, :, -1], current_value)
        assert torch.all(view.valid[:, -1])
        assert torch.all(view.positions[:, -1] == position)
        assert_empty_slots_have_zero_position(view)

        read_positions = list(range(max(1, position - window_size + 1), position + 1))
        for batch in range(BATCH_SIZE):
            assert view.positions[batch, view.valid[batch]].tolist() == read_positions
            assert int(view.valid[batch].sum()) == len(read_positions)
            valid_keys = view.key[batch][:, view.valid[batch], :]
            valid_values = view.value[batch][:, view.valid[batch], :]
            for slot, old_position in enumerate(read_positions):
                assert torch.equal(valid_keys[:, slot], inserted[old_position][0][batch])
                assert torch.equal(valid_values[:, slot], inserted[old_position][1][batch])

        updated = cache.update(current_key, current_value, position=position)
        retained_positions = list(
            range(max(1, position - window_size + 2), position + 1)
        )
        assert updated.next_position == position + 1
        assert updated.key.shape == updated.value.shape == (
            BATCH_SIZE, NUM_HEADS, window_size - 1, HEAD_DIM
        )
        assert updated.valid.shape == updated.positions.shape == (
            BATCH_SIZE, window_size - 1
        )
        assert_empty_slots_have_zero_position(updated)
        for batch in range(BATCH_SIZE):
            assert updated.positions[batch, updated.valid[batch]].tolist() == retained_positions
            assert int(updated.valid[batch].sum()) == len(retained_positions)
            assert int(updated.valid[batch].sum()) <= window_size - 1
            for slot, old_position in enumerate(retained_positions):
                stored_slot = updated.key[batch][:, updated.valid[batch], :][:, slot]
                stored_value = updated.value[batch][:, updated.valid[batch], :][:, slot]
                assert torch.equal(stored_slot, inserted[old_position][0][batch])
                assert torch.equal(stored_value, inserted[old_position][1][batch])

        if window_size == 1:
            assert updated.key.shape[2] == updated.value.shape[2] == 0
            assert updated.valid.numel() == updated.positions.numel() == 0
            assert view.valid.shape[1] == 1
        cache = updated


@pytest.mark.parametrize("window_size", [2, 3, 5])
def test_update_matches_read_with_oldest_slot_dropped(window_size):
    cache = empty_cache(window_size)
    for position in range(1, LAST_POSITION + 1):
        current_key, current_value = make_entry()
        view = cache.read(current_key, current_value, position=position)
        updated = cache.update(current_key, current_value, position=position)

        assert torch.equal(updated.key, view.key[:, :, 1:])
        assert torch.equal(updated.value, view.value[:, :, 1:])
        assert torch.equal(updated.valid, view.valid[:, 1:])
        assert torch.equal(updated.positions, view.positions[:, 1:])
        cache = updated


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_read_is_pure_and_rejects_wrong_position(window_size):
    cache = empty_cache(window_size)
    first_key, first_value = make_entry()
    cache = cache.update(first_key, first_value, position=1)
    current_key, current_value = make_entry()
    old_key = cache.key.clone()
    old_value = cache.value.clone()
    old_valid = cache.valid.clone()
    old_positions = cache.positions.clone()
    old_next_position = cache.next_position

    first = cache.read(current_key, current_value, position=2)
    second = cache.read(current_key, current_value, position=2)

    assert_view_equal(first, second)
    assert torch.equal(cache.key, old_key)
    assert torch.equal(cache.value, old_value)
    assert torch.equal(cache.valid, old_valid)
    assert torch.equal(cache.positions, old_positions)
    assert cache.next_position == old_next_position == 2
    for invalid_position in (1, 3):
        with pytest.raises(ValueError):
            cache.read(current_key, current_value, position=invalid_position)


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_update_does_not_modify_previous_cache(window_size):
    cache = empty_cache(window_size)
    for position in range(1, LAST_POSITION + 1):
        old_key = cache.key.clone()
        old_value = cache.value.clone()
        old_valid = cache.valid.clone()
        old_positions = cache.positions.clone()
        old_next_position = cache.next_position
        current_key, current_value = make_entry()

        updated = cache.update(current_key, current_value, position=position)

        assert torch.equal(cache.key, old_key)
        assert torch.equal(cache.value, old_value)
        assert torch.equal(cache.valid, old_valid)
        assert torch.equal(cache.positions, old_positions)
        assert cache.next_position == old_next_position == position
        cache = updated


@pytest.mark.parametrize("window_size", [2, 3, 5])
def test_retained_entries_keep_gradients_across_later_updates(window_size):
    cache = empty_cache(window_size)
    inserted = {}
    for position in range(1, LAST_POSITION + 1):
        current_key, current_value = make_entry()
        current_key.requires_grad_()
        current_value.requires_grad_()
        inserted[position] = (current_key, current_value)
        cache = cache.update(current_key, current_value, position=position)

        retained_positions = range(max(1, position - window_size + 2), position + 1)
        for old_position in retained_positions:
            old_key, old_value = inserted[old_position]
            key_gradient = torch.autograd.grad(
                cache.key.sum(), old_key, retain_graph=True
            )[0]
            value_gradient = torch.autograd.grad(
                cache.value.sum(), old_value, retain_graph=True
            )[0]
            assert torch.count_nonzero(key_gradient) > 0
            assert torch.count_nonzero(value_gradient) > 0


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_update_rejects_repeated_and_skipped_positions(window_size):
    cache = empty_cache(window_size)
    current_key, current_value = make_entry()
    with pytest.raises(ValueError):
        cache.update(current_key, current_value, position=2)

    cache = cache.update(current_key, current_value, position=1)
    next_key, next_value = make_entry()
    for invalid_position in (1, 3):
        with pytest.raises(ValueError):
            cache.update(next_key, next_value, position=invalid_position)


@pytest.mark.parametrize("window_size", WINDOW_SIZES)
def test_batch_sequences_are_isolated(window_size):
    original = empty_cache(window_size)
    changed = empty_cache(window_size)

    for position in range(1, LAST_POSITION + 1):
        current_key, current_value = make_entry()
        changed_key = current_key.clone()
        changed_value = current_value.clone()
        if position == 3:
            changed_key[0] += 10.0
            changed_value[0] -= 10.0

        original_view = original.read(current_key, current_value, position=position)
        changed_view = changed.read(changed_key, changed_value, position=position)
        assert torch.equal(original_view.key[1], changed_view.key[1])
        assert torch.equal(original_view.value[1], changed_view.value[1])
        assert torch.equal(original_view.valid[1], changed_view.valid[1])
        assert torch.equal(original_view.positions[1], changed_view.positions[1])
        if position == 3:
            assert not torch.equal(original_view.key[0], changed_view.key[0])
            assert not torch.equal(original_view.value[0], changed_view.value[0])

        original = original.update(current_key, current_value, position=position)
        changed = changed.update(changed_key, changed_value, position=position)
        assert torch.equal(original.key[1], changed.key[1])
        assert torch.equal(original.value[1], changed.value[1])
        assert torch.equal(original.valid[1], changed.valid[1])
        assert torch.equal(original.positions[1], changed.positions[1])
        if position == 3 and window_size > 1:
            assert not torch.equal(original.key[0], changed.key[0])
            assert not torch.equal(original.value[0], changed.value[0])
