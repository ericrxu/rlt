"""Experiment 5 contracts for independent, equal-sized length groups."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import train as train_module
from compare import compare_results
from train import TrainConfig, load_config, make_training_batch, train


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MIX = [4, 8, 16, 32]


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2053)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


@pytest.fixture
def config_data():
    data = json.loads((CONFIG_DIR / "parity_rlt_final.json").read_text(encoding="utf-8"))
    return data | {
        "name": "length_mix_test", "length_mix": None,
        "dim": 4, "num_encoder_layers": 1, "num_decoder_layers": 1,
        "num_heads": 1, "head_dim": 4, "window_size": 2,
        "train_length": 2, "batch_size": 4, "train_programs": None,
        "steps": 1, "warmup_steps": 1,
        "eval_lengths": [2], "eval_programs": 4, "eval_interval": 1,
    }


def test_length_mix_is_required_and_null_is_valid(config_data, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data), encoding="utf-8")
    assert load_config(path).length_mix is None

    missing = config_data.copy()
    del missing["length_mix"]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="length_mix"):
        load_config(path)


@pytest.mark.parametrize("mix,other", [
    ([], {}),
    ("1,2", {}),
    ({"lengths": [1, 2]}, {}),
    ((1, 2), {}),
    ([0, 2], {}),
    ([-1, 2], {}),
    ([1, 2.0], {}),
    ([True, 2], {}),
    ([1, 1, 2], {}),
    ([2, 1], {}),
    ([1], {}),  # largest length must equal train_length
    ([1, 2], {"batch_size": 3}),
    ([1, 2], {"train_programs": 4}),
    ([1, 2], {"curriculum": [[2, 1]]}),
])
def test_invalid_length_mix_raises_value_error(config_data, mix, other):
    with pytest.raises(ValueError):
        TrainConfig(**(config_data | other | {"length_mix": mix}))


def test_mixed_training_groups_are_equal_sized_and_step_deterministic(config_data):
    config = TrainConfig(**(config_data | {
        "train_length": 32, "batch_size": 64, "length_mix": MIX,
    }))
    first = train_module.make_training_groups(config, 1)
    repeat = train_module.make_training_groups(config, 1)
    next_step = train_module.make_training_groups(config, 2)

    assert len(first) == len(repeat) == len(next_step) == 4
    for length, (tokens, labels), same in zip(MIX, first, repeat):
        assert tokens.shape == (16, length + 1)
        assert labels.shape == (16, length)
        assert all(torch.equal(left, right) for left, right in zip((tokens, labels), same))
    assert any(
        not torch.equal(tokens, later_tokens)
        for (tokens, _), (later_tokens, _) in zip(first, next_step)
    )


def test_null_length_mix_returns_exact_existing_training_batch(config_data):
    config = TrainConfig(**config_data)
    groups = train_module.make_training_groups(config, 1)
    old_batch = make_training_batch(config, 1)
    assert len(groups) == 1
    assert all(torch.equal(left, right) for left, right in zip(groups[0], old_batch))


def test_mixed_step_loss_is_mean_final_state_loss_over_all_programs(
    config_data, tmp_path, monkeypatch,
):
    config = TrainConfig(**(config_data | {"length_mix": [1, 2]}))
    groups = train_module.make_training_groups(config, 1)
    captured_logits = []
    real_build_model = train_module.build_model

    def capture_model(model_config):
        model = real_build_model(model_config)
        model.register_forward_hook(
            lambda _module, _inputs, logits: captured_logits.append(logits.detach().clone())
        )
        return model

    monkeypatch.setattr(train_module, "build_model", capture_model)
    monkeypatch.setattr(train_module, "evaluate", lambda _model, _config: {})
    result = train(config, output_dir=tmp_path / "results",
                   checkpoint_dir=tmp_path / "checkpoints", dtype=torch.float64)

    assert len(captured_logits) == len(groups) == 2
    expected = F.cross_entropy(
        torch.cat([logits[:, -1] for logits in captured_logits]),
        torch.cat([labels[:, -1] for _tokens, labels in groups]),
    )
    assert result.loss_history[0] == pytest.approx(expected.item(), abs=1e-12)


@pytest.mark.parametrize("group_index", (0, -1), ids=("shortest", "longest"))
def test_each_group_final_labels_change_training_gradients(
    config_data, tmp_path, monkeypatch, group_index,
):
    config = TrainConfig(**(config_data | {"length_mix": [1, 2]}))
    real_groups = train_module.make_training_groups
    real_clip = train_module.clip_gradients

    def run(changed, directory):
        seen_groups = []
        gradients = []

        def groups_for_step(model_config, step):
            groups = real_groups(model_config, step)
            if changed:
                tokens, labels = groups[group_index]
                labels = labels.clone()
                labels[:, -1] = 1 - labels[:, -1]
                groups[group_index] = (tokens, labels)
            seen_groups.extend((tokens.clone(), labels.clone()) for tokens, labels in groups)
            return groups

        def capture_gradients(model, limit):
            gradients.append({
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            })
            return real_clip(model, limit)

        with monkeypatch.context() as patch:
            patch.setattr(train_module, "make_training_groups", groups_for_step)
            patch.setattr(train_module, "clip_gradients", capture_gradients)
            patch.setattr(train_module, "evaluate", lambda _model, _config: {})
            train(config, output_dir=directory / "results",
                  checkpoint_dir=directory / "checkpoints", dtype=torch.float64)
        assert len(gradients) == 1
        return seen_groups, gradients[0]

    original_groups, original_gradients = run(False, tmp_path / "original")
    changed_groups, changed_gradients = run(True, tmp_path / "changed")
    assert len(original_groups) == len(changed_groups) == 2
    for index, ((original_tokens, original_labels),
                (changed_tokens, changed_labels)) in enumerate(
                    zip(original_groups, changed_groups)
                ):
        assert torch.equal(original_tokens, changed_tokens)
        assert torch.equal(original_labels[:, :-1], changed_labels[:, :-1])
        if index == group_index % len(original_groups):
            assert not torch.equal(original_labels[:, -1], changed_labels[:, -1])
        else:
            assert torch.equal(original_labels[:, -1], changed_labels[:, -1])
    assert original_gradients.keys() == changed_gradients.keys()
    assert any(
        not torch.equal(original_gradients[name], changed_gradients[name])
        for name in original_gradients
    )


def test_tiny_mixed_training_run_records_length_mix(config_data, tmp_path):
    config = TrainConfig(**(config_data | {"length_mix": [1, 2]}))
    result = train(config, output_dir=tmp_path / "results",
                   checkpoint_dir=tmp_path / "checkpoints", dtype=torch.float64)
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert len(result.loss_history) == 1
    assert payload["config"]["length_mix"] == [1, 2]
    assert payload["loss_history"] == result.loss_history


def _write_result(directory, model_type, mix, *, omit_mix=False):
    config = json.loads((CONFIG_DIR / f"parity_{model_type}_final.json").read_text(
        encoding="utf-8"
    ))
    config["length_mix"] = mix
    if omit_mix:
        del config["length_mix"]
    payload = {
        "config": config, "git_commit": "a" * 40, "dirty": False,
        "seed": config["seed"],
        "eval_history": [{"step": config["steps"], "lengths": {
            str(length): {"final_state_accuracy": 0.5}
            for length in config["eval_lengths"]
        }}],
    }
    (directory / f"{model_type}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("left,right,omit_left", [
    ([4, 8, 16, 32], [8, 16, 32], False),
    (None, [4, 8, 16, 32], False),
    (None, [4, 8, 16, 32], True),
])
def test_compare_rejects_different_length_mixes(tmp_path, left, right, omit_left):
    _write_result(tmp_path, "rlt", left, omit_mix=omit_left)
    _write_result(tmp_path, "gru", right)
    with pytest.raises(ValueError, match="length_mix"):
        compare_results(tmp_path, expected_seeds=(0,))


def test_compare_treats_missing_length_mix_as_null(tmp_path):
    _write_result(tmp_path, "rlt", None, omit_mix=True)
    _write_result(tmp_path, "gru", None)
    assert len(compare_results(tmp_path, expected_seeds=(0,))) == 2


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("model_type", ("rlt", "gru"))
def test_experiment_5_config_matches_final_state_reference(task, model_type):
    final_path = CONFIG_DIR / f"{task}_{model_type}_final.json"
    mixed_path = CONFIG_DIR / f"{task}_{model_type}_mixed_final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    mixed = json.loads(mixed_path.read_text(encoding="utf-8"))
    assert final["length_mix"] is None
    assert mixed == final | {
        "name": f"{task}_{model_type}_mixed_final",
        "length_mix": MIX,
    }
    assert json.loads(json.dumps(asdict(load_config(mixed_path)))) == mixed


def test_all_other_configs_explicitly_use_null_length_mix():
    for path in CONFIG_DIR.glob("*.json"):
        if path.stem.endswith("_mixed_final"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "length_mix" in data and data["length_mix"] is None, path.name
