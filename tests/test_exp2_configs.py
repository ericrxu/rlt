"""Experiment 2 configs differ from Experiment 1 only in name and objective."""

import json
from pathlib import Path

import pytest

from train import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("model_type", ("rlt", "gru"))
def test_final_state_config_only_changes_name_and_objective(task, model_type):
    reference_path = CONFIG_DIR / f"{task}_{model_type}.json"
    final_path = CONFIG_DIR / f"{task}_{model_type}_final.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))

    assert reference["objective"] == "per_position"
    assert final == reference | {
        "name": f"{task}_{model_type}_final",
        "objective": "final_state",
    }
    assert load_config(reference_path).objective == "per_position"
    assert load_config(final_path).objective == "final_state"


def test_existing_configs_explicitly_use_per_position():
    for path in CONFIG_DIR.glob("*.json"):
        if path.stem.endswith("_final"):
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["objective"] == "per_position", path.name
