"""Experiment 3 curriculum contracts. Training steps are one-based."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from compare import compare_results
from train import TrainConfig, load_config, make_training_batch, train


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CURRICULUM = [[4, 1500], [8, 1500], [16, 1500], [32, 1500]]


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2051)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


@pytest.fixture
def config_data():
    data = json.loads((CONFIG_DIR / "parity_rlt_final.json").read_text(encoding="utf-8"))
    return data | {
        "name": "curriculum_test", "curriculum": None,
        "dim": 4, "num_encoder_layers": 1, "num_decoder_layers": 1,
        "num_heads": 1, "head_dim": 4, "window_size": 2,
        "train_length": 2, "batch_size": 4, "train_programs": None,
        "steps": 2, "warmup_steps": 1,
        "eval_lengths": [2], "eval_programs": 4, "eval_interval": 2,
    }


def test_curriculum_is_required_and_null_is_valid(config_data, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config_data), encoding="utf-8")
    assert load_config(path).curriculum is None

    missing = config_data.copy()
    del missing["curriculum"]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="curriculum"):
        load_config(path)


@pytest.mark.parametrize("curriculum,other", [
    ([], {}),
    ("two stages", {}),
    ({"length": 2, "steps": 2}, {}),
    ([[1, 1]], {}),  # stage steps do not sum to the run length
    ([[1, 1], [2, 2]], {}),
    ([[1, 1], [3, 1]], {}),  # final length differs from train_length
    ([[0, 1], [2, 1]], {}),
    ([[1, 1], [-2, 1]], {}),
    ([[3, 1], [2, 1]], {}),
    ([[1, 0], [2, 2]], {}),
    ([[1, -1], [2, 3]], {}),
    ([[1, 1, 1], [2, 1]], {}),
    ([[1, 1], [2]], {}),
    ([[1, 1], [2, 1]], {"train_programs": 4}),
])
def test_invalid_curriculum_raises_value_error(config_data, curriculum, other):
    with pytest.raises(ValueError):
        TrainConfig(**(config_data | other | {"curriculum": curriculum}))


@pytest.mark.parametrize("step,length", [
    (1, 4), (1500, 4), (1501, 8), (3000, 8),
    (3001, 16), (4500, 16), (4501, 32), (6000, 32),
])
def test_training_batch_uses_current_stage_at_exact_boundaries(
    config_data, step, length,
):
    config = TrainConfig(**(config_data | {
        "train_length": 32, "steps": 6000, "curriculum": CURRICULUM,
    }))
    tokens, labels = make_training_batch(config, step)
    assert tokens.shape == (config.batch_size, length + 1)
    assert labels.shape == (config.batch_size, length)


def test_single_stage_curriculum_preserves_batches_and_loss_history(config_data, tmp_path):
    without = TrainConfig(**config_data)
    with_stage = TrainConfig(**(config_data | {"curriculum": [[2, 2]]}))
    for step in (1, 2):
        old_batch = make_training_batch(without, step)
        stage_batch = make_training_batch(with_stage, step)
        assert all(torch.equal(a, b) for a, b in zip(old_batch, stage_batch))

    old_run = train(without, output_dir=tmp_path / "old_results",
                    checkpoint_dir=tmp_path / "old_checkpoints", dtype=torch.float64)
    stage_run = train(with_stage, output_dir=tmp_path / "stage_results",
                      checkpoint_dir=tmp_path / "stage_checkpoints", dtype=torch.float64)
    assert old_run.loss_history == stage_run.loss_history


def test_two_stage_training_completes_and_records_curriculum(config_data, tmp_path):
    curriculum = [[1, 1], [2, 1]]
    config = TrainConfig(**(config_data | {"curriculum": curriculum}))
    result = train(config, output_dir=tmp_path / "results",
                   checkpoint_dir=tmp_path / "checkpoints", dtype=torch.float64)
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert len(result.loss_history) == 2
    assert payload["config"]["curriculum"] == curriculum
    assert payload["loss_history"] == result.loss_history


def _result(directory, name, curriculum, *, model_type="rlt", omit_curriculum=False):
    config = json.loads((CONFIG_DIR / f"parity_{model_type}_final.json").read_text(
        encoding="utf-8"
    ))
    config.update(name=name, curriculum=curriculum)
    if omit_curriculum:
        del config["curriculum"]
    payload = {
        "config": config, "git_commit": "a" * 40, "dirty": False,
        "seed": config["seed"],
        "eval_history": [{"step": config["steps"], "lengths": {
            str(length): {"final_state_accuracy": 0.5}
            for length in config["eval_lengths"]
        }}],
    }
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("left,right,omit_left", [
    ([[16, 1500], [32, 1500]], [[8, 1500], [32, 1500]], False),
    (None, [[16, 1500], [32, 1500]], True),
])
def test_compare_rejects_mixed_curricula(tmp_path, left, right, omit_left):
    _result(tmp_path, "old", left, omit_curriculum=omit_left)
    _result(tmp_path, "new", right, model_type="gru")
    with pytest.raises(ValueError, match="curriculum"):
        compare_results(tmp_path, expected_seeds=(0,))


def test_compare_treats_missing_curriculum_as_null(tmp_path):
    _result(tmp_path, "legacy", None, omit_curriculum=True)
    _result(tmp_path, "explicit_null", None, model_type="gru")
    assert len(compare_results(tmp_path, expected_seeds=(0,))) == 2


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("model_type", ("rlt", "gru"))
def test_experiment_3_config_matches_final_state_reference(task, model_type):
    final_path = CONFIG_DIR / f"{task}_{model_type}_final.json"
    curriculum_path = CONFIG_DIR / f"{task}_{model_type}_curriculum.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    assert final["curriculum"] is None
    assert curriculum == final | {
        "name": f"{task}_{model_type}_curriculum",
        "steps": 6000,
        "curriculum": CURRICULUM,
    }
    assert json.loads(json.dumps(asdict(load_config(curriculum_path))))["curriculum"] == CURRICULUM


def test_all_other_configs_explicitly_use_null_curriculum():
    for path in CONFIG_DIR.glob("*.json"):
        if path.stem.endswith("_curriculum"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "curriculum" in data and data["curriculum"] is None, path.name
