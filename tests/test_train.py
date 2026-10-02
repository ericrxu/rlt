"""Contract for the Stage 9b training runner.

Steps are one-based throughout: step 1 uses the first warmup learning rate,
and logits[:, i + 1] score the state after operation i.
"""

import ast
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest
import torch

import train as train_module
from objectives import state_tracking_accuracy, state_tracking_loss
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
        name="parity_test", task="parity", model_type="rlt",
        objective="per_position", seed=17, eval_seed=701,
        dim=4, num_encoder_layers=1, num_decoder_layers=1,
        num_heads=1, head_dim=4, window_size=2, alpha=0.75,
        rms_eps=1e-6, train_length=2, batch_size=16, train_programs=16,
        curriculum=None,
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


def test_config_requires_objective_and_rejects_unknown_value(config, tmp_path):
    data = asdict(config)
    path = tmp_path / "config.json"

    missing = data.copy()
    del missing["objective"]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_config(path)

    path.write_text(json.dumps(data | {"objective": "unknown"}), encoding="utf-8")
    with pytest.raises(ValueError, match="objective"):
        load_config(path)


def _one_step_with_labels(config, tmp_path, monkeypatch, labels):
    tokens = torch.tensor([[BOS, 0, 1, 0], [BOS, 1, 0, 1]])
    logits_seen = []
    gradients_seen = []
    original_build_model = train_module.build_model
    original_clip_gradients = train_module.clip_gradients

    def capture_model(run_config):
        model = original_build_model(run_config)
        model.register_forward_hook(
            lambda _module, _inputs, logits: logits_seen.append(logits.detach().clone())
        )
        return model

    def capture_gradients(model, limit):
        gradients_seen.append({
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        })
        return original_clip_gradients(model, limit)

    with monkeypatch.context() as patch:
        patch.setattr(train_module, "build_model", capture_model)
        patch.setattr(train_module, "clip_gradients", capture_gradients)
        patch.setattr(train_module, "make_training_batch",
                      lambda _config, _step: (tokens, labels))
        patch.setattr(train_module, "_git_metadata", lambda: ("a" * 40, False))
        result = train(
            replace(config, train_length=3, batch_size=2, train_programs=2,
                    steps=1, warmup_steps=1, eval_interval=1),
            output_dir=tmp_path / "results", dtype=torch.float64,
            checkpoint_dir=tmp_path / "checkpoints",
        )
    assert len(gradients_seen) == 1
    return result.loss_history[0], logits_seen[0], gradients_seen[0]


@pytest.mark.parametrize("objective,expect_equal", [
    ("final_state", True), ("per_position", False),
])
def test_training_gradients_respond_to_labels_selected_by_objective(
    config, tmp_path, monkeypatch, objective, expect_equal,
):
    first_labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    changed_labels = torch.tensor([[1, 0, 0], [0, 1, 1]])
    run_config = replace(config, objective=objective)

    _, first_logits, first_grads = _one_step_with_labels(
        run_config, tmp_path / "first", monkeypatch, first_labels,
    )
    _, changed_logits, changed_grads = _one_step_with_labels(
        run_config, tmp_path / "changed", monkeypatch, changed_labels,
    )
    assert torch.equal(first_logits, changed_logits)
    assert first_grads.keys() == changed_grads.keys()
    equal = all(torch.equal(first_grads[name], changed_grads[name])
                for name in first_grads)
    assert equal is expect_equal


def test_per_position_step_loss_matches_state_tracking_loss(
    config, tmp_path, monkeypatch,
):
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    actual, logits, _ = _one_step_with_labels(
        config, tmp_path, monkeypatch, labels,
    )
    expected = state_tracking_loss(logits, labels).item()
    assert actual == expected


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
    first = train(config, output_dir=tmp_path / "first", dtype=torch.float64,
                  checkpoint_dir=tmp_path / "first" / "checkpoints")
    second = train(config, output_dir=tmp_path / "second", dtype=torch.float64,
                   checkpoint_dir=tmp_path / "second" / "checkpoints")
    different = train(
        replace(config, seed=config.seed + 1),
        output_dir=tmp_path / "different", dtype=torch.float64,
        checkpoint_dir=tmp_path / "different" / "checkpoints",
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
    result = train(overfit, output_dir=tmp_path, dtype=torch.float64,
                   checkpoint_dir=tmp_path / "checkpoints")
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
    result = train(config, output_dir=tmp_path, dtype=torch.float64,
                   checkpoint_dir=tmp_path / "checkpoints")
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


def test_fixed_training_accuracy_matches_final_model_and_fresh_mode_omits_it(
    config, tmp_path,
):
    fixed = replace(config, eval_interval=1)
    result = train(fixed, output_dir=tmp_path / "fixed", dtype=torch.float64,
                   checkpoint_dir=tmp_path / "fixed_checkpoints")
    tokens, labels = make_training_batch(fixed, 1)
    with torch.no_grad():
        position, final = state_tracking_accuracy(result.model(tokens), labels)
    assert all("training_set" in entry for entry in result.eval_history)
    assert result.eval_history[-1]["training_set"] == {
        "per_position_accuracy": position.item(),
        "final_state_accuracy": final.item(),
    }
    saved = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert saved["eval_history"] == result.eval_history

    fresh = replace(config, train_programs=None, eval_interval=1)
    fresh_result = train(fresh, output_dir=tmp_path / "fresh", dtype=torch.float64,
                         checkpoint_dir=tmp_path / "fresh_checkpoints")
    assert all("training_set" not in entry for entry in fresh_result.eval_history)


def test_invalid_prediction_rate_uses_only_state_label_positions():
    parity_logits = torch.zeros((2, 3, 3), dtype=torch.float64)
    parity_logits[:, 1:, 2] = 1
    assert train_module.invalid_prediction_rate(parity_logits, "parity") == 1.0

    parity_logits[:, 1:, 2] = 0
    parity_logits[:, 1, 0] = 1
    parity_logits[:, 2, 1] = 1
    parity_logits[:, 0, 2] = 1  # BOS is not a state-label prediction.
    assert train_module.invalid_prediction_rate(parity_logits, "parity") == 0.0

    parity_logits[0, 2, 1] = 0
    parity_logits[0, 2, 2] = 2
    assert train_module.invalid_prediction_rate(parity_logits, "parity") == 0.25

    five_state_logits = torch.zeros((2, 3, 5), dtype=torch.float64)
    five_state_logits[:, 1:, 4] = 1
    assert train_module.invalid_prediction_rate(five_state_logits, "five_state") == 0.0


@pytest.mark.parametrize("task,outputs,chosen_class,expected", [
    ("parity", 3, 2, 1.0),
    ("five_state", 5, 4, 0.0),
])
def test_evaluate_reports_invalid_prediction_rate_for_each_length(
    config, task, outputs, chosen_class, expected,
):
    class ConstantPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

        def forward(self, tokens):
            logits = torch.zeros((*tokens.shape, outputs), dtype=torch.float64,
                                 device=tokens.device)
            logits[:, 1:, chosen_class] = 1
            return logits

    lengths = (1, 3)
    metrics = evaluate(ConstantPrediction(), replace(config, task=task,
                                                     eval_lengths=lengths))
    assert set(metrics) == {str(length) for length in lengths}
    for length in lengths:
        assert metrics[str(length)]["invalid_prediction_rate"] == expected


def _progress_lines(output):
    return [line for line in output.splitlines() if line.strip()]


def test_progress_printing_preserves_training_and_obeys_interval(
    config, tmp_path, capsys,
):
    run_config = replace(config, steps=4, eval_interval=2)

    def run(name, log_every):
        result = train(run_config, output_dir=tmp_path / name, dtype=torch.float64,
                       checkpoint_dir=tmp_path / name / "checkpoints",
                       log_every=log_every)
        return result, _progress_lines(capsys.readouterr().out)

    baseline, silent_lines = run("silent", None)
    every_step, step_lines = run("each_step", 1)
    every_second, alternate_lines = run("alternate", 2)
    assert silent_lines == []
    for logged in (every_step, every_second):
        assert logged.loss_history == baseline.loss_history
        assert logged.eval_history == baseline.eval_history
        _assert_same_parameters(_parameters(logged.model), _parameters(baseline.model))
    assert len(step_lines) == 4
    assert len(alternate_lines) == 2
    for lines, steps in ((step_lines, (1, 2, 3, 4)),
                         (alternate_lines, (2, 4))):
        for line, step in zip(lines, steps):
            assert re.search(rf"\bstep\s*(?:=|:)?\s*{step}\b", line, re.I)
            assert re.search(r"\bloss\b\s*(?:=|:)?\s*[-+\d.eE]+", line, re.I)
            assert re.search(r"(?:pre.?clip.?grad.?norm|grad.?norm)\s*(?:=|:)?\s*[-+\d.eE]+",
                             line, re.I)
            assert re.search(r"(?:\blr\b|learning.?rate)\s*(?:=|:)?\s*[-+\d.eE]+",
                             line, re.I)
            assert re.search(r"(?:mean.?gate|\bgate\b)\s*(?:=|:)?\s*[-+\d.eE]+",
                             line, re.I)


def test_cli_forwards_log_every(config, tmp_path, monkeypatch, capsys):
    source = ast.parse(Path(train_module.__file__).read_text(encoding="utf-8"))
    main_block = source.body[-1]
    assert isinstance(main_block, ast.If)
    calls = []
    sentinel = object()

    def fake_train(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(results_path=tmp_path / "result.json")

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(tmp_path / "config.json"),
                                       "--log-every", "2"])
    namespace = {
        "__doc__": train_module.__doc__,
        "Path": Path,
        "load_config": lambda path: sentinel,
        "train": fake_train,
    }
    exec(compile(ast.Module(body=main_block.body, type_ignores=[]),
                 train_module.__file__, "exec"), namespace)
    assert len(calls) == 1
    assert calls[0][0] == (sentinel,)
    assert calls[0][1]["log_every"] == 2
    assert capsys.readouterr().out.strip() == str(tmp_path / "result.json")
