"""Contract for the Stage 9b training runner.

Steps are one-based throughout: step 1 uses the first warmup learning rate,
and logits[:, i + 1] score the state after operation i.
"""

from dataclasses import asdict, replace
import json
import math

import pytest
import torch

from task import BOS
from train import (
    TrainConfig,
    build_model,
    clip_gradients,
    count_parameters,
    diagnostic_hooks,
    evaluate,
    learning_rate_at_step,
    load_config,
    make_eval_batch,
    make_optimizer,
    make_training_batch,
    train,
)


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2029)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


@pytest.fixture
def config():
    return TrainConfig(
        name="parity_test", task="parity", seed=17, eval_seed=701,
        dim=4, num_encoder_layers=1, num_decoder_layers=1,
        num_heads=1, head_dim=4, window_size=2, alpha=0.75,
        rms_eps=1e-6, train_length=2, batch_size=16, train_programs=16,
        steps=2, learning_rate=0.01, min_learning_rate=0.001,
        warmup_steps=1, weight_decay=0.01, grad_clip_norm=0.5,
        eval_lengths=(2,), eval_programs=4, eval_interval=2,
    )


def _parameters(model):
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _assert_same_parameters(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _assert_same_batch(left, right):
    assert len(left) == len(right) == 2
    for a, b in zip(left, right):
        assert torch.equal(a, b)


def test_config_requires_every_field_and_rejects_unknown_fields(config, tmp_path):
    data = asdict(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert json.loads(json.dumps(asdict(load_config(path)))) == json.loads(json.dumps(data))

    missing = data.copy()
    del missing["alpha"]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_config(path)

    unknown = data | {"surprise": 1}
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_config(path)


def test_fixed_program_count_must_match_batch_size(config, tmp_path):
    data = asdict(config)
    data["batch_size"] = data["train_programs"] - 1
    path = tmp_path / "mismatched_batch.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_direct_config_construction_rejects_mismatched_batch_size(config):
    data = asdict(config)
    data["batch_size"] = data["train_programs"] - 1
    with pytest.raises(ValueError):
        TrainConfig(**data)


@pytest.mark.parametrize("task,outputs", [("parity", 3), ("five_state", 5)])
def test_model_output_size_and_parameter_count(config, task, outputs):
    model = build_model(replace(config, task=task))
    tokens = torch.tensor([[BOS, 0, 1]])
    assert model(tokens).shape == (1, 3, outputs)
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


def test_training_is_bit_identical_for_same_seed_and_changes_for_new_seed(
    config, tmp_path,
):
    first = train(config, output_dir=tmp_path / "first", dtype=torch.float64)
    second = train(config, output_dir=tmp_path / "second", dtype=torch.float64)
    different = train(
        replace(config, seed=config.seed + 1),
        output_dir=tmp_path / "different", dtype=torch.float64,
    )
    assert first.loss_history == second.loss_history
    _assert_same_parameters(_parameters(first.model), _parameters(second.model))
    assert first.loss_history != different.loss_history
    assert any(
        not torch.equal(a, b)
        for a, b in zip(first.model.parameters(), different.model.parameters())
    )


def test_fixed_sixteen_programs_can_overfit_in_fifty_steps(config, tmp_path):
    overfit = replace(
        config, train_length=1, batch_size=16, train_programs=16,
        steps=50, learning_rate=0.03, min_learning_rate=0.03,
        warmup_steps=1, eval_interval=50,
    )
    result = train(overfit, output_dir=tmp_path, dtype=torch.float64)
    assert len(result.loss_history) == 50
    assert result.loss_history[-1] < result.loss_history[0] / 2


def test_training_batches_are_keyed_by_seed_and_step(config):
    fixed_a = make_training_batch(config, step=1)
    fixed_b = make_training_batch(config, step=9)
    _assert_same_batch(fixed_a, fixed_b)
    assert not torch.equal(
        fixed_a[0], make_training_batch(replace(config, seed=31), step=1)[0]
    )

    fresh = replace(config, train_programs=None, train_length=12, batch_size=8)
    _assert_same_batch(
        make_training_batch(fresh, step=4), make_training_batch(fresh, step=4)
    )
    assert not torch.equal(
        make_training_batch(fresh, step=4)[0],
        make_training_batch(fresh, step=5)[0],
    )


def test_eval_at_training_length_uses_different_programs(config):
    sample = replace(
        config, train_length=32, eval_lengths=(32,), eval_programs=16,
    )
    training_tokens, _ = make_training_batch(sample, step=1)
    eval_tokens, _ = make_eval_batch(sample, length=32)
    assert training_tokens.shape == eval_tokens.shape
    for eval_program in eval_tokens:
        assert all(not torch.equal(eval_program, training_program)
                   for training_program in training_tokens)


def test_eval_programs_ignore_training_seed_and_eval_length_list(config):
    base = replace(config, train_length=12, eval_lengths=(12, 16))
    changed_seed = replace(base, seed=999)
    extra_length = replace(base, eval_lengths=(8, 12, 16))
    for length in base.eval_lengths:
        expected = make_eval_batch(base, length)
        _assert_same_batch(expected, make_eval_batch(changed_seed, length))
        _assert_same_batch(expected, make_eval_batch(extra_length, length))


def test_evaluation_consumes_the_same_programs_for_each_value_key(config):
    model = build_model(config)

    def observed(config_to_evaluate):
        seen = {}

        def capture(_module, args):
            tokens = args[0]
            length = tokens.shape[1] - 1
            seen.setdefault(length, []).append(tokens.detach().clone())

        handle = model.register_forward_pre_hook(capture)
        try:
            evaluate(model, config_to_evaluate)
        finally:
            handle.remove()
        return seen

    base = replace(config, eval_lengths=(2, 3))
    changed_seed = replace(base, seed=999)
    extra_length = replace(base, eval_lengths=(1, 2, 3))
    first = observed(base)
    second = observed(changed_seed)
    expanded = observed(extra_length)
    assert set(first) == {2, 3}
    for length in first:
        assert len(first[length]) == len(second[length]) == len(expanded[length])
        for a, b, c in zip(first[length], second[length], expanded[length]):
            assert torch.equal(a, b)
            assert torch.equal(a, c)


def test_diagnostic_hooks_leave_logits_unchanged_and_are_removed(config):
    model = build_model(config)
    tokens = torch.tensor([[BOS, 1, 0], [BOS, 0, 1]])
    expected = model(tokens)
    with diagnostic_hooks(model) as diagnostics:
        actual = model(tokens)
        assert torch.equal(actual, expected)
        assert len(model.merge._forward_hooks) == 1
        assert len(model.decoder._forward_hooks) == 1
        assert len(diagnostics.gate_values) == tokens.shape[1]
        assert len(diagnostics.state_norms) == tokens.shape[1]
    assert not model.merge._forward_hooks
    assert not model.decoder._forward_hooks
    assert torch.equal(model(tokens), expected)


def test_diagnostics_record_gates_and_one_state_norm_per_position(config):
    model = build_model(config)
    tokens = torch.tensor([[BOS, 0, 1, 1]])
    with diagnostic_hooks(model) as diagnostics:
        model(tokens)
    assert len(diagnostics.gate_values) == tokens.shape[1]
    assert len(diagnostics.state_norms) == tokens.shape[1]
    assert all(torch.all((gate > 0) & (gate < 1)) for gate in diagnostics.gate_values)
    assert all(torch.isfinite(norm).all() and torch.all(norm > 0)
               for norm in diagnostics.state_norms)


def test_clipping_logs_true_preclip_norm_and_enforces_limit(config):
    model = build_model(config)
    tokens = torch.tensor([[BOS, 1, 0]])
    model(tokens).square().sum().backward()
    true_norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad.detach())
        for p in model.parameters() if p.grad is not None
    ])).item()
    limit = true_norm / 3
    logged = clip_gradients(model, limit)
    assert math.isclose(logged, true_norm, rel_tol=1e-12, abs_tol=1e-12)
    after = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad.detach())
        for p in model.parameters() if p.grad is not None
    ])).item()
    assert after <= limit + 1e-12


def test_weight_decay_groups_follow_tensor_rank(config):
    model = build_model(config)
    optimizer = make_optimizer(model, config)
    rates = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups for parameter in group["params"]
    }
    assert len(rates) == len(list(model.parameters()))
    for name, parameter in model.named_parameters():
        expected = config.weight_decay if parameter.ndim >= 2 else 0.0
        assert rates[id(parameter)] == expected, name
    assert rates[id(model.s_star)] == 0.0


def test_evaluation_does_not_change_parameters(config):
    model = build_model(config)
    before = _parameters(model)
    evaluate(model, config)
    _assert_same_parameters(before, _parameters(model))


def test_results_file_contains_reproducibility_and_accuracy_fields(config, tmp_path):
    result = train(config, output_dir=tmp_path, dtype=torch.float64)
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert payload["config"] == json.loads(json.dumps(asdict(config)))
    assert isinstance(payload["git_commit"], str) and len(payload["git_commit"]) == 40
    assert isinstance(payload["dirty"], bool)
    assert payload["seed"] == config.seed
    assert payload["dtype"] == "float64"
    assert isinstance(payload["torch_version"], str)
    assert payload["parameter_count"] == sum(p.numel() for p in result.model.parameters())
    assert len(payload["loss_history"]) == config.steps
    assert set(map(int, payload["eval_history"][-1]["lengths"])) == set(config.eval_lengths)
    for metrics in payload["eval_history"][-1]["lengths"].values():
        assert 0 <= metrics["final_state_accuracy"] <= 1


def test_learning_rate_warmup_and_cosine_endpoints(config):
    scheduled = replace(
        config, steps=12, warmup_steps=3,
        learning_rate=0.02, min_learning_rate=0.002,
    )
    rates = [learning_rate_at_step(scheduled, step) for step in range(1, 13)]
    assert rates[0] > 0
    assert math.isclose(rates[0], scheduled.learning_rate / 3, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(rates[2], scheduled.learning_rate, rel_tol=0, abs_tol=1e-15)
    assert all(next_rate <= rate for rate, next_rate in zip(rates[2:], rates[3:]))
    assert math.isclose(rates[-1], scheduled.min_learning_rate,
                        rel_tol=0, abs_tol=1e-15)
