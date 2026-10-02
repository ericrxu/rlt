"""Experiment 1 config contracts for Stage 11a."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from train import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
TASKS = ("parity", "five_state")
VARIANTS = ("rlt", "rlt_alpha0", "transformer", "gru")
SHARED_FIELDS = (
    "seed", "eval_seed", "train_length", "batch_size", "train_programs",
    "steps", "learning_rate", "min_learning_rate", "warmup_steps",
    "weight_decay", "grad_clip_norm", "eval_lengths", "eval_programs",
    "eval_interval",
)
EXPERIMENT_1 = {
    "seed": 0,
    "eval_seed": 1000,
    "train_length": 32,
    "batch_size": 64,
    "train_programs": None,
    "steps": 3000,
    "learning_rate": 0.003,
    "min_learning_rate": 0.0003,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "eval_lengths": [32, 48, 64, 96, 128],
    "eval_programs": 2048,
    "eval_interval": 1000,
}


def _raw_config(task, variant):
    path = CONFIG_DIR / f"{task}_{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("task", TASKS)
def test_experiment_1_configs_share_exact_training_and_eval_settings(task):
    configs = {variant: _raw_config(task, variant) for variant in VARIANTS}
    for variant, data in configs.items():
        assert {field: data[field] for field in SHARED_FIELDS} == EXPERIMENT_1, variant
        assert asdict(load_config(CONFIG_DIR / f"{task}_{variant}.json"))["task"] == task


@pytest.mark.parametrize("task", TASKS)
def test_alpha_zero_config_only_changes_alpha_and_name(task):
    reference = _raw_config(task, "rlt")
    zero = _raw_config(task, "rlt_alpha0")
    assert reference["name"] == f"{task}_rlt"
    assert zero["name"] == f"{task}_rlt_alpha0"
    assert reference["alpha"] == 0.5
    assert zero["alpha"] == 0.0
    assert {key: value for key, value in zero.items() if key not in ("name", "alpha")} == {
        key: value for key, value in reference.items() if key not in ("name", "alpha")
    }
