"""Experiment 4 checkpoint and configuration contracts."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

import train as train_module
from train import build_model, evaluate, load_config, train


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2052)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


@pytest.fixture
def short_config():
    return replace(
        load_config(CONFIG_DIR / "parity_rlt_curriculum.json"),
        name="exp4_checkpoint_test", seed=17, eval_seed=701,
        dim=4, num_encoder_layers=1, num_decoder_layers=1,
        num_heads=1, head_dim=4, window_size=2,
        train_length=2, batch_size=4, curriculum=None,
        steps=4, warmup_steps=1, eval_lengths=(2,),
        eval_programs=4, eval_interval=2,
    )


def _state_dict(model):
    return {name: value.detach().clone()
            for name, value in model.state_dict().items()}


def _assert_same_state_dict(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_eval_checkpoints_and_final_checkpoint_replay(short_config, tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    evaluated_states = []
    real_evaluate = train_module.evaluate

    def record_evaluation(model, config):
        evaluated_states.append(_state_dict(model))
        return real_evaluate(model, config)

    with monkeypatch.context() as patch:
        patch.setattr(train_module, "evaluate", record_evaluation)
        result = train(
            short_config, output_dir=tmp_path / "results",
            checkpoint_dir=checkpoint_dir, dtype=torch.float64,
        )

    final_path = result.checkpoint_path
    stamp = final_path.name.removesuffix(f"_{short_config.name}.pt")
    assert final_path.name == f"{stamp}_{short_config.name}.pt"
    expected_paths = {
        checkpoint_dir / f"{stamp}_{short_config.name}_step2.pt": 2,
        checkpoint_dir / f"{stamp}_{short_config.name}_step4.pt": 4,
        final_path: 4,
    }
    assert set(checkpoint_dir.glob("*.pt")) == set(expected_paths)
    assert [entry["step"] for entry in result.eval_history] == [2, 4]
    assert len(evaluated_states) == 2
    assert any(not torch.equal(evaluated_states[0][name], evaluated_states[1][name])
               for name in evaluated_states[0])

    for path, step in expected_paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["config"] == asdict(short_config)
        assert checkpoint["step"] == step
        assert checkpoint["model"].keys() == result.model.state_dict().keys()
        if path != final_path:
            _assert_same_state_dict(checkpoint["model"], evaluated_states[step // 2 - 1])

    step_checkpoint = torch.load(
        checkpoint_dir / f"{stamp}_{short_config.name}_step2.pt",
        map_location="cpu", weights_only=True,
    )
    restored = build_model(short_config).to(dtype=torch.float64)
    restored.load_state_dict(step_checkpoint["model"])
    assert evaluate(restored, short_config) == result.eval_history[0]["lengths"]

    final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=True)
    _assert_same_state_dict(final_checkpoint["model"], _state_dict(result.model))


def test_eval_interval_does_not_change_training(short_config, tmp_path):
    runs = {}
    for interval in (1, 2, 4):
        config = replace(short_config, eval_interval=interval)
        runs[interval] = train(
            config, output_dir=tmp_path / str(interval) / "results",
            checkpoint_dir=tmp_path / str(interval) / "checkpoints",
            dtype=torch.float64,
        )

    reference = runs[4]
    reference_state = _state_dict(reference.model)
    for interval in (1, 2):
        run = runs[interval]
        assert run.loss_history == reference.loss_history
        _assert_same_state_dict(_state_dict(run.model), reference_state)
        assert run.eval_history[-1]["lengths"] == reference.eval_history[-1]["lengths"]


@pytest.mark.parametrize("task", ("parity", "five_state"))
@pytest.mark.parametrize("arm,alpha", (("a05", 0.5), ("a01", 0.1)))
def test_experiment_4_config_changes_only_name_interval_and_alpha(task, arm, alpha):
    base_path = CONFIG_DIR / f"{task}_rlt_curriculum.json"
    experiment_path = CONFIG_DIR / f"{task}_rlt_{arm}_curriculum.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))

    expected = base | {
        "name": f"{task}_rlt_{arm}_curriculum",
        "eval_interval": 1500,
    }
    if arm == "a01":
        expected["alpha"] = 0.1
    assert experiment == expected
    assert experiment["alpha"] == alpha
    assert json.loads(json.dumps(asdict(load_config(experiment_path)))) == experiment
