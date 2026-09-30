"""Contracts for the whole-sequence and incremental RLT transition."""

import pytest
import torch

from model import RLTModel


VOCAB_SIZE = 31
DIM = 8
HEADS = 2
HEAD_DIM = 4
BOS = 2
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


def make_model(alpha=0.75, window_size=3):
    return RLTModel(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        window_size=window_size,
        alpha=alpha,
        rms_eps=EPS,
        bos_token_id=BOS,
        num_memory_groups=1,
    )


def test_num_memory_groups_is_required():
    with pytest.raises(TypeError):
        RLTModel(
            vocab_size=VOCAB_SIZE,
            dim=DIM,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_heads=HEADS,
            head_dim=HEAD_DIM,
            window_size=3,
            alpha=0.75,
            rms_eps=EPS,
            bos_token_id=BOS,
        )


def make_tokens(batch_size=2, length=12):
    tokens = torch.randint(3, VOCAB_SIZE, (batch_size, length))
    tokens[:, 0] = BOS
    return tokens


def initial_state(model, tokens):
    return model.initial_state(tokens.shape[0], tokens.device)


def snapshot_state(state):
    return (
        state.next_position,
        state.s.clone(),
        state.encoder_cache.next_position,
        tuple(
            (layer.key.clone(), layer.value.clone())
            for layer in state.encoder_cache.layers
        ),
        state.memory_cache.positions.clone(),
        tuple(
            (group.key.clone(), group.value.clone(), group.positions.clone())
            for group in state.memory_cache.groups
        ),
        tuple(
            (
                cache.key.clone(), cache.value.clone(), cache.valid.clone(),
                cache.positions.clone(), cache.next_position,
            )
            for cache in state.decoder_caches
        ),
    )


def assert_state_matches_snapshot(state, snapshot):
    (
        next_position, s, encoder_next_position, encoder_layers,
        memory_positions, memory_groups, decoder_caches,
    ) = snapshot
    assert state.next_position == next_position
    assert torch.equal(state.s, s)
    assert state.encoder_cache.next_position == encoder_next_position
    assert len(state.encoder_cache.layers) == len(encoder_layers)
    for layer, (key, value) in zip(state.encoder_cache.layers, encoder_layers):
        assert torch.equal(layer.key, key)
        assert torch.equal(layer.value, value)
    assert torch.equal(state.memory_cache.positions, memory_positions)
    assert len(state.memory_cache.groups) == len(memory_groups)
    for group, (key, value, positions) in zip(
        state.memory_cache.groups, memory_groups
    ):
        assert torch.equal(group.key, key)
        assert torch.equal(group.value, value)
        assert torch.equal(group.positions, positions)
    assert len(state.decoder_caches) == len(decoder_caches)
    for cache, (key, value, valid, positions, cache_next_position) in zip(
        state.decoder_caches, decoder_caches
    ):
        assert torch.equal(cache.key, key)
        assert torch.equal(cache.value, value)
        assert torch.equal(cache.valid, valid)
        assert torch.equal(cache.positions, positions)
        assert cache.next_position == cache_next_position


def test_forward_matches_reference_loop_through_own_submodules():
    model = make_model()
    tokens = make_tokens()
    batch_size, length = tokens.shape
    positions = torch.arange(1, length + 1).expand(batch_size, -1)
    features = model.encoder(tokens, positions)
    memory_cache = model.memory(features, positions)
    s = model.s_star.expand(batch_size, -1)
    decoder_caches = model.decoder.empty_caches(batch_size, tokens.device)
    expected = []

    for position in range(1, length + 1):
        u, _ = model.merge(features[:, position - 1], s)
        s, decoder_caches = model.decoder(
            u, position, model.memory, memory_cache, decoder_caches
        )
        expected.append(model.readout(model.norm_o(s)))

    expected = torch.stack(expected, dim=1)
    actual = model(tokens)
    assert actual.shape == (batch_size, length, VOCAB_SIZE)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("alpha,should_change", [(0.75, True), (0.0, False)])
def test_initial_state_enters_first_update_only_through_merge(alpha, should_change):
    model = make_model(alpha=alpha)
    tokens = make_tokens(batch_size=1, length=1)
    original = model(tokens)

    with torch.no_grad():
        model.s_star.copy_(torch.arange(1, DIM + 1, dtype=torch.float64))
    changed = model(tokens)

    if should_change:
        assert not torch.equal(original[:, 0], changed[:, 0])
    else:
        assert torch.equal(original, changed)


@pytest.mark.parametrize("alpha,should_reach", [(0.75, True), (0.0, False)])
def test_recurrence_reaches_twelfth_token_when_window_is_one(alpha, should_reach):
    model = make_model(alpha=alpha, window_size=1)
    tokens = make_tokens(batch_size=1, length=12)
    last_logits = model(tokens)[:, -1]
    gradient = torch.autograd.grad(
        last_logits.square().sum(), model.s_star, allow_unused=True
    )[0]

    if should_reach:
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0
    elif gradient is not None:
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_changing_token_cannot_change_earlier_logits():
    model = make_model()
    tokens = make_tokens(batch_size=1)
    changed_tokens = tokens.clone()
    changed_index = 5
    changed_tokens[0, changed_index] = (
        (tokens[0, changed_index] - 3 + 1) % (VOCAB_SIZE - 3) + 3
    )

    original = model(tokens)
    changed = model(changed_tokens)
    assert torch.equal(original[:, :changed_index], changed[:, :changed_index])
    assert not torch.equal(original[:, changed_index], changed[:, changed_index])


def test_step_matches_forward_at_every_position():
    model = make_model()
    tokens = make_tokens()
    expected = model(tokens)
    state = initial_state(model, tokens)
    outputs = []

    for index in range(tokens.shape[1]):
        logits, state = model.step(tokens[:, index], state)
        outputs.append(logits)

    actual = torch.stack(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert state.next_position == tokens.shape[1] + 1


def test_forward_is_bit_identical_to_prefill_from_fresh_state():
    model = make_model()
    tokens = make_tokens()
    expected = model(tokens)
    actual, state = model.prefill(tokens, initial_state(model, tokens))

    assert torch.equal(actual, expected)
    assert state.next_position == tokens.shape[1] + 1


def test_prefix_prefill_then_steps_matches_forward():
    model = make_model()
    tokens = make_tokens()
    expected = model(tokens)
    prefix, state = model.prefill(tokens[:, :5], initial_state(model, tokens))
    outputs = [prefix]

    for index in range(5, 12):
        logits, state = model.step(tokens[:, index], state)
        outputs.append(logits[:, None, :])

    actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert state.next_position == 13


def test_two_prefill_chunks_match_forward():
    model = make_model()
    tokens = make_tokens()
    expected = model(tokens)
    first, state = model.prefill(tokens[:, :5], initial_state(model, tokens))
    second, state = model.prefill(tokens[:, 5:], state)

    actual = torch.cat((first, second), dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert state.next_position == 13


def test_step_reuses_state_without_mutating_it():
    model = make_model()
    tokens = make_tokens()
    _, state = model.prefill(tokens[:, :5], initial_state(model, tokens))
    original = snapshot_state(state)

    first_logits, first_state = model.step(tokens[:, 5], state)
    assert_state_matches_snapshot(state, original)
    second_logits, second_state = model.step(tokens[:, 5], state)
    assert torch.equal(first_logits, second_logits)
    assert_state_matches_snapshot(first_state, snapshot_state(second_state))
    assert_state_matches_snapshot(state, original)


def test_prefill_reuses_state_without_mutating_it():
    model = make_model()
    tokens = make_tokens()
    _, state = model.prefill(tokens[:, :5], initial_state(model, tokens))
    original = snapshot_state(state)

    first_logits, first_state = model.prefill(tokens[:, 5:], state)
    assert_state_matches_snapshot(state, original)
    second_logits, second_state = model.prefill(tokens[:, 5:], state)
    assert torch.equal(first_logits, second_logits)
    assert_state_matches_snapshot(first_state, snapshot_state(second_state))
    assert_state_matches_snapshot(state, original)


def test_fresh_sequence_requires_bos_for_step_forward_and_prefill():
    model = make_model()
    tokens = make_tokens(batch_size=2)
    tokens[1, 0] = 3

    with pytest.raises(ValueError):
        model.step(tokens[:, 0], initial_state(model, tokens))
    with pytest.raises(ValueError):
        model(tokens)
    with pytest.raises(ValueError):
        model.prefill(tokens[:, :5], initial_state(model, tokens))


def test_last_position_loss_reaches_every_parameter_tensor():
    model = make_model(alpha=0.75, window_size=3)
    tokens = make_tokens()
    logits = model(tokens)
    logits[:, -1].square().sum().backward()

    parameters = dict(model.named_parameters())
    assert parameters
    assert any(name.startswith("encoder.") for name in parameters)
    assert any(name.startswith("memory.") for name in parameters)
    assert any(name.startswith("merge.") for name in parameters)
    assert any(name.startswith("decoder.") for name in parameters)
    assert any(name.startswith("norm_o.") for name in parameters)
    assert any(name.startswith("readout.") for name in parameters)
    assert "s_star" in parameters
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_batch_sequences_have_isolated_logits():
    model = make_model()
    tokens = make_tokens()
    changed_tokens = tokens.clone()
    changed_index = 5
    changed_tokens[0, changed_index] = (
        (tokens[0, changed_index] - 3 + 1) % (VOCAB_SIZE - 3) + 3
    )

    original = model(tokens)
    changed = model(changed_tokens)
    assert torch.equal(original[1], changed[1])
    assert not torch.equal(original[0, changed_index], changed[0, changed_index])
