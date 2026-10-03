"""Experiment 6 contracts: one full batch per one-based cycle step."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

import train as train_module
from compare import compare_results
from train import TrainConfig, load_config, make_training_batch, train


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CYCLE = [4, 8, 16, 32]


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2054)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


@pytest.fixture
def config_data():
    data = json.loads((CONFIG_DIR / "parity_rlt_final.json").read_text(encoding="utf-8"))
    return data | {
        "name": "length_cycle_test", "length_cycle": None,
        "curriculum": None, "length_mix": None,
        "dim": 4, "num_encoder_layers": 1, "num_decoder_layers": 1,
        "num_heads": 1, "head_dim": 4, "window_size": 2,
        "train_length": 2, "batch_size": 3, "train_programs": None,
        "steps": 4, "warmup_steps": 1,
        "eval_lengths": [2], "eval_programs": 3, "eval_interval": 4,
    }


def test_length_cycle_is_required(config_data, tmp_path):
    missing = config_data.copy()
    del missing["length_cycle"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="length_cycle"):
        load_config(path)


def test_null_length_cycle_is_valid(config_data, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data), encoding="utf-8")
    assert load_config(path).length_cycle is None


@pytest.mark.parametrize("cycle,other", [
    ([], {}),
    ("1,2", {}),
    ({"lengths": [1, 2]}, {}),
    ((1, 2), {}),
    (True, {}),
    (False, {}),
    (2, {}),
    (2.0, {}),
    ([0, 2], {}),
    ([-1, 2], {}),
    ([1, 2.0], {}),
    ([1, "2"], {}),
    ([1, None], {}),
    ([True, 2], {}),
    ([False, 2], {}),
    ([1, 1, 2], {}),
    ([2, 1], {}),
    ([1], {}),
    ([1, 3], {}),
    ([1, 2], {"train_programs": 3}),
    ([1, 2], {"curriculum": [[1, 2], [2, 2]]}),
    ([1, 2], {"length_mix": [1, 2], "batch_size": 4}),
])
def test_invalid_length_cycle_raises_value_error(config_data, cycle, other):
    with pytest.raises(ValueError, match="length_cycle"):
        TrainConfig(**(config_data | other | {"length_cycle": cycle}))


def test_cycle_does_not_require_batch_size_divisible_by_cycle_size(config_data):
    config = TrainConfig(**(config_data | {"length_cycle": [1, 2]}))
    assert config.batch_size == 3


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("step,length", [
    (1, 4), (2, 8), (3, 16), (4, 32),
    (5, 4), (6, 8), (7, 16), (8, 32), (3000, 32),
])
def test_cycle_batches_use_full_batch_and_unchanged_step_seed(
    config_data, task, step, length,
):
    config = TrainConfig(**(config_data | {
        "task": task, "train_length": 32, "steps": 3000,
        "batch_size": 64, "length_cycle": CYCLE,
    }))
    tokens, labels = make_training_batch(config, step)
    assert tokens.shape == (64, length + 1)
    assert labels.shape == (64, length)
    reference = make_training_batch(
        replace(config, train_length=length, length_cycle=None), step,
    )
    assert torch.equal(tokens, reference[0])
    assert torch.equal(labels, reference[1])


@pytest.mark.parametrize("step", (0, -1))
def test_cycle_batches_reject_nonpositive_steps(config_data, step):
    config = TrainConfig(**(config_data | {"length_cycle": [1, 2]}))
    with pytest.raises(ValueError, match="one-based"):
        make_training_batch(config, step)


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("model_type", ("rlt", "gru"))
def test_single_length_cycle_preserves_batches_and_loss_history(
    config_data, tmp_path, task, model_type,
):
    data = config_data | {"task": task, "model_type": model_type}
    if model_type == "gru":
        data |= dict.fromkeys((
            "num_decoder_layers", "num_heads", "head_dim", "window_size",
            "alpha", "rms_eps",
        ))
    without = TrainConfig(**data)
    with_cycle = replace(without, length_cycle=[without.train_length])
    for step in range(1, without.steps + 1):
        old_tokens, old_labels = make_training_batch(without, step)
        cycle_tokens, cycle_labels = make_training_batch(with_cycle, step)
        assert torch.equal(old_tokens, cycle_tokens)
        assert torch.equal(old_labels, cycle_labels)

    old_run = train(without, output_dir=tmp_path / "old_results",
                    checkpoint_dir=tmp_path / "old_checkpoints", dtype=torch.float64)
    cycle_run = train(with_cycle, output_dir=tmp_path / "cycle_results",
                      checkpoint_dir=tmp_path / "cycle_checkpoints", dtype=torch.float64)
    assert len(old_run.loss_history) == len(cycle_run.loss_history) == without.steps
    assert old_run.loss_history == cycle_run.loss_history


def test_tiny_cycle_training_completes_and_records_cycle(
    config_data, tmp_path, monkeypatch,
):
    config = TrainConfig(**(config_data | {"length_cycle": [1, 2]}))
    seen_shapes = []
    real_batch = train_module.make_training_batch

    def record_batch(model_config, step):
        tokens, labels = real_batch(model_config, step)
        seen_shapes.append((step, tokens.shape, labels.shape))
        return tokens, labels

    monkeypatch.setattr(train_module, "make_training_batch", record_batch)
    result = train(config, output_dir=tmp_path / "results",
                   checkpoint_dir=tmp_path / "checkpoints", dtype=torch.float64)
    assert seen_shapes == [
        (step, (3, length + 1), (3, length))
        for step, length in enumerate((1, 2, 1, 2), start=1)
    ]
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert len(result.loss_history) == 4
    assert payload["config"]["length_cycle"] == [1, 2]
    assert payload["loss_history"] == result.loss_history
    assert result.eval_history[-1]["step"] == 4
    assert result.checkpoint_path.is_file()


def _write_result(directory, name, model_type, cycle, *, omit_cycle=False):
    config = json.loads((CONFIG_DIR / f"parity_{model_type}_final.json").read_text(
        encoding="utf-8"
    ))
    config.update(name=name, length_cycle=cycle)
    if omit_cycle:
        del config["length_cycle"]
    payload = {
        "config": config, "git_commit": "a" * 40, "dirty": False,
        "seed": config["seed"],
        "eval_history": [{"step": config["steps"], "lengths": {
            str(length): {"final_state_accuracy": 0.5}
            for length in config["eval_lengths"]
        }}],
    }
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("reverse_order", (False, True))
@pytest.mark.parametrize("left,right,omit_left", [
    (CYCLE, [8, 16, 32], False),
    (None, CYCLE, False),
    (None, CYCLE, True),
])
def test_compare_rejects_different_length_cycles(
    tmp_path, left, right, omit_left, reverse_order,
):
    left_name, right_name = ("b", "a") if reverse_order else ("a", "b")
    _write_result(tmp_path, left_name, "rlt", left, omit_cycle=omit_left)
    _write_result(tmp_path, right_name, "gru", right)
    with pytest.raises(ValueError, match="length_cycle"):
        compare_results(tmp_path, expected_seeds=(0,))


@pytest.mark.parametrize("reverse_order", (False, True))
def test_compare_treats_missing_length_cycle_as_null(tmp_path, reverse_order):
    legacy_name, null_name = ("b", "a") if reverse_order else ("a", "b")
    _write_result(tmp_path, legacy_name, "rlt", None, omit_cycle=True)
    _write_result(tmp_path, null_name, "gru", None)
    assert len(compare_results(tmp_path, expected_seeds=(0,))) == 2


def test_compare_accepts_matching_non_null_cycles(tmp_path):
    _write_result(tmp_path, "a", "rlt", CYCLE)
    _write_result(tmp_path, "b", "gru", CYCLE)
    assert len(compare_results(tmp_path, expected_seeds=(0,))) == 2


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("model_type", ("rlt", "gru"))
def test_experiment_6_config_matches_final_state_reference(task, model_type):
    final_path = CONFIG_DIR / f"{task}_{model_type}_final.json"
    cycle_path = CONFIG_DIR / f"{task}_{model_type}_cycle_final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    cycled = json.loads(cycle_path.read_text(encoding="utf-8"))
    assert final["length_cycle"] is None
    assert cycled == final | {
        "name": f"{task}_{model_type}_cycle_final",
        "length_cycle": CYCLE,
    }
    assert json.loads(json.dumps(asdict(load_config(cycle_path)))) == cycled


def test_all_other_configs_explicitly_use_null_length_cycle():
    cycle_names = {
        f"{task}_{model_type}_cycle_final"
        for task in ("parity", "five_state") for model_type in ("rlt", "gru")
    }
    for path in CONFIG_DIR.glob("*.json"):
        if path.stem in cycle_names:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "length_cycle" in data and data["length_cycle"] is None, path.name
