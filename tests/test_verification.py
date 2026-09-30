"""Whole-model consequences of RLT Proposition 3.1 and Appendices B and C."""

from dataclasses import replace

import pytest
import torch

from model import RLTModel


VOCAB = 31
BOS = 2
L_D = 2


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(8137)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


def make_model(*, window=3, alpha=0.75, groups=1):
    return RLTModel(
        vocab_size=VOCAB,
        dim=8,
        num_encoder_layers=2,
        num_decoder_layers=L_D,
        num_heads=2,
        head_dim=4,
        window_size=window,
        alpha=alpha,
        rms_eps=1e-6,
        bos_token_id=BOS,
        num_memory_groups=groups,
    )


def tokens12(batch=1):
    tokens = torch.randint(3, VOCAB, (batch, 12))
    tokens[:, 0] = BOS
    return tokens


def assert_tensor(actual, expected, *, exact=False, row=None):
    if row is not None:
        actual, expected = actual[row], expected[row]
    assert actual.shape == expected.shape
    if exact:
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)


def assert_state(actual, expected, *, exact=False, row=None):
    """Compare every tensor and every position counter in an inference snapshot."""
    assert actual.next_position == expected.next_position
    assert_tensor(actual.s, expected.s, exact=exact, row=row)

    assert actual.encoder_cache.next_position == expected.encoder_cache.next_position
    assert len(actual.encoder_cache.layers) == len(expected.encoder_cache.layers)
    for got, want in zip(actual.encoder_cache.layers, expected.encoder_cache.layers):
        assert_tensor(got.key, want.key, exact=exact, row=row)
        assert_tensor(got.value, want.value, exact=exact, row=row)

    assert_tensor(
        actual.memory_cache.positions, expected.memory_cache.positions,
        exact=True, row=row,
    )
    assert len(actual.memory_cache.groups) == len(expected.memory_cache.groups)
    for got, want in zip(actual.memory_cache.groups, expected.memory_cache.groups):
        assert_tensor(got.key, want.key, exact=exact, row=row)
        assert_tensor(got.value, want.value, exact=exact, row=row)
        assert_tensor(got.positions, want.positions, exact=True, row=row)

    assert len(actual.decoder_caches) == len(expected.decoder_caches) == L_D
    for got, want in zip(actual.decoder_caches, expected.decoder_caches):
        assert_tensor(got.key, want.key, exact=exact, row=row)
        assert_tensor(got.value, want.value, exact=exact, row=row)
        assert_tensor(got.valid, want.valid, exact=True, row=row)
        assert_tensor(got.positions, want.positions, exact=True, row=row)
        assert got.next_position == want.next_position


def step_all(model, tokens):
    state = model.initial_state(tokens.shape[0], tokens.device)
    logits, states = [], []
    for token in tokens.unbind(dim=1):
        output, state = model.step(token, state)
        logits.append(output)
        states.append(state)
    return torch.stack(logits, dim=1), states


def detach_decoder_caches(state):
    return [replace(cache, key=cache.key.detach(), value=cache.value.detach())
            for cache in state.decoder_caches]


def clone_state(state):
    return replace(
        state,
        s=state.s.clone(),
        encoder_cache=replace(
            state.encoder_cache,
            layers=tuple(replace(layer, key=layer.key.clone(), value=layer.value.clone())
                         for layer in state.encoder_cache.layers),
        ),
        memory_cache=replace(
            state.memory_cache,
            positions=state.memory_cache.positions.clone(),
            groups=tuple(replace(group, key=group.key.clone(), value=group.value.clone(),
                                 positions=group.positions.clone())
                         for group in state.memory_cache.groups),
        ),
        decoder_caches=[replace(cache, key=cache.key.clone(), value=cache.value.clone(),
                                valid=cache.valid.clone(), positions=cache.positions.clone())
                        for cache in state.decoder_caches],
    )


@pytest.mark.parametrize("window", [1, 3])
@pytest.mark.parametrize("alpha", [0.0, 0.75])
@pytest.mark.parametrize("groups", [1, L_D])
def test_serving_split_invariance(window, alpha, groups):
    model = make_model(window=window, alpha=alpha, groups=groups)
    tokens = tokens12()
    with torch.no_grad():
        full_logits, full_state = model.prefill(tokens, model.initial_state(1, tokens.device))
        step_logits, step_states = step_all(model, tokens)
        assert_tensor(step_logits, full_logits)
        assert_state(step_states[-1], full_state)

        for k in range(1, 12):
            _, split_state = model.prefill(tokens[:, :k], model.initial_state(1, tokens.device))
            assert_state(split_state, step_states[k - 1])
            continuation = []
            for token in tokens[:, k:].unbind(dim=1):
                logits, split_state = model.step(token, split_state)
                continuation.append(logits)
            assert_tensor(torch.stack(continuation, dim=1), full_logits[:, k:])
            assert_state(split_state, full_state)

        chunks_state = model.initial_state(1, tokens.device)
        chunks = []
        for start in (0, 4, 8):
            logits, chunks_state = model.prefill(tokens[:, start:start + 4], chunks_state)
            chunks.append(logits)
        assert_tensor(torch.cat(chunks, dim=1), full_logits)
        assert_state(chunks_state, full_state)


def test_full_state_causality():
    model = make_model()
    original = tokens12()
    changed = original.clone()
    j = 6  # Absolute position; BOS is position 1.
    changed[0, j - 1] = 3 + (changed[0, j - 1] - 2) % (VOCAB - 3)
    with torch.no_grad():
        _, original_states = step_all(model, original)
        _, changed_states = step_all(model, changed)
    for i in range(1, j):
        assert_state(original_states[i - 1], changed_states[i - 1], exact=True)
    assert not torch.equal(original_states[j - 1].s, changed_states[j - 1].s)


def test_decoder_state_has_two_gradient_paths():
    tokens = tokens12()
    weights = torch.randn(VOCAB)
    model = make_model(window=3, alpha=0.75)
    _, prefix = model.prefill(tokens[:, :2], model.initial_state(1, tokens.device))
    variants = {
        "intact": prefix,
        "s_only": replace(prefix, s=prefix.s.detach()),
        "cache_only": replace(prefix, decoder_caches=detach_decoder_caches(prefix)),
        "both": replace(prefix, s=prefix.s.detach(),
                        decoder_caches=detach_decoder_caches(prefix)),
    }
    outputs = {}
    for name, boundary in variants.items():
        logits, _ = model.step(tokens[:, 2], boundary)
        outputs[name] = logits
        grad = torch.autograd.grad((logits * weights).sum(), model.s_star,
                                   allow_unused=True, retain_graph=True)[0]
        if name == "both":
            assert grad is None or torch.count_nonzero(grad) == 0
        else:
            assert grad is not None and torch.count_nonzero(grad) > 0, name
    for logits in outputs.values():
        assert_tensor(logits, outputs["intact"], exact=True)

    # With W=1 no historical decoder KV exists, so detaching s cuts this path.
    model_w1 = make_model(window=1, alpha=0.75)
    _, prefix_w1 = model_w1.prefill(tokens[:, :2], model_w1.initial_state(1, tokens.device))
    intact_w1, _ = model_w1.step(tokens[:, 2], prefix_w1)
    logits_w1, _ = model_w1.step(tokens[:, 2], replace(prefix_w1, s=prefix_w1.s.detach()))
    assert_tensor(logits_w1, intact_w1, exact=True)
    intact_grad_w1 = torch.autograd.grad((intact_w1 * weights).sum(), model_w1.s_star,
                                         allow_unused=True, retain_graph=True)[0]
    assert intact_grad_w1 is not None and torch.count_nonzero(intact_grad_w1) > 0
    grad_w1 = torch.autograd.grad((logits_w1 * weights).sum(), model_w1.s_star,
                                  allow_unused=True)[0]
    assert grad_w1 is None or torch.count_nonzero(grad_w1) == 0


def test_selected_finite_difference_gradients():
    model = make_model(window=3, alpha=0.75)
    tokens = tokens12()[:, :4]
    weights = torch.randn(1, 4, VOCAB)

    def loss():
        return (model(tokens) * weights).sum()

    selected = (
        model.s_star,
        model.merge.state.weight,
        model.decoder.layers[0].value.weight,
        model.encoder.layers[0].query.weight,
    )
    analytic = torch.autograd.grad(loss(), selected)
    h = 1e-5
    for parameter, gradient in zip(selected, analytic):
        for flat in range(3):
            with torch.no_grad():
                original = parameter.view(-1)[flat].item()
                parameter.view(-1)[flat] = original + h
            plus = loss().item()
            with torch.no_grad():
                parameter.view(-1)[flat] = original - h
            minus = loss().item()
            with torch.no_grad():
                parameter.view(-1)[flat] = original
            numerical = (plus - minus) / (2 * h)
            torch.testing.assert_close(
                gradient.view(-1)[flat], torch.tensor(numerical), rtol=1e-6, atol=1e-9,
            )


def test_snapshot_reuse_for_two_continuations():
    model = make_model()
    tokens = tokens12()
    alternate = tokens.clone()
    alternate[0, 7] = 3 + (alternate[0, 7] - 2) % (VOCAB - 3)
    with torch.no_grad():
        _, saved = model.prefill(tokens[:, :5], model.initial_state(1, tokens.device))
        snapshot = clone_state(saved)
        for full_tokens in (tokens, alternate):
            state = saved
            suffix_logits = []
            for token in full_tokens[:, 5:].unbind(dim=1):
                logits, state = model.step(token, state)
                suffix_logits.append(logits)
            assert_tensor(torch.stack(suffix_logits, dim=1), model(full_tokens)[:, 5:])
            assert_state(saved, snapshot, exact=True)


def test_full_state_batch_isolation():
    model = make_model()
    original = tokens12(batch=2)
    changed = original.clone()
    j = 6
    changed[0, j - 1] = 3 + (changed[0, j - 1] - 2) % (VOCAB - 3)
    with torch.no_grad():
        _, original_states = step_all(model, original)
        _, changed_states = step_all(model, changed)
    for i, (got, want) in enumerate(zip(original_states, changed_states), start=1):
        assert_state(got, want, exact=True, row=1)
        if i >= j:
            assert not torch.equal(got.s[0], want.s[0])


def test_forward_is_deterministic():
    model = make_model()
    tokens = tokens12(batch=2)
    with torch.no_grad():
        assert_tensor(model(tokens), model(tokens), exact=True)
