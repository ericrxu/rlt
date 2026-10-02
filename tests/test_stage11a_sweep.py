"""Stage 11a sweep contracts. All training stays in train.py."""

from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import train as train_module
from train import load_config, train


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2041)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


def _tiny_config(tmp_path):
    config = replace(
        load_config(CONFIG_DIR / "parity_tiny.json"),
        name="parity_sweep_test", train_length=2, batch_size=4,
        train_programs=4, steps=2, warmup_steps=1,
        eval_lengths=(2,), eval_programs=4, eval_interval=2,
    )
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")
    return path


def _run(sweep, config_path, tmp_path, seeds=(0, 1), cpu_threads=2):
    return sweep.run_sweep(
        config_path, seeds=seeds, cpu_threads=cpu_threads,
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
    )


def test_train_records_actual_cpu_thread_count_in_every_result(tmp_path, monkeypatch):
    config = replace(load_config(CONFIG_DIR / "parity_tiny.json"),
                     train_length=2, batch_size=4, train_programs=4,
                     steps=1, warmup_steps=1, eval_lengths=(2,),
                     eval_programs=4, eval_interval=1)
    monkeypatch.setattr(train_module, "_git_metadata", lambda: (COMMIT, False))
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = train(config, output_dir=tmp_path / "results", dtype=torch.float64,
                       checkpoint_dir=tmp_path / "checkpoints")
        payload = json.loads(result.results_path.read_text(encoding="utf-8"))
        assert payload["cpu_threads"] == torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(old_threads)


def test_sweep_overrides_seed_and_appends_seed_to_name(tmp_path, monkeypatch):
    sweep = importlib.import_module("run_sweep")
    config_path = _tiny_config(tmp_path)
    monkeypatch.setattr(train_module, "_git_metadata", lambda: (COMMIT, False))
    seen = []

    def fake_train(config, **kwargs):
        seen.append((config.seed, config.name, torch.get_num_threads()))
        return SimpleNamespace(results_path=tmp_path / f"{config.name}.json")

    monkeypatch.setattr(sweep, "train", fake_train)
    old_threads = torch.get_num_threads()
    try:
        _run(sweep, config_path, tmp_path, seeds=(0, 2), cpu_threads=2)
        assert seen == [
            (0, "parity_sweep_test_s0", 2),
            (2, "parity_sweep_test_s2", 2),
        ]
    finally:
        torch.set_num_threads(old_threads)


def test_sweep_refuses_dirty_repo_before_training(tmp_path, monkeypatch):
    sweep = importlib.import_module("run_sweep")
    config_path = _tiny_config(tmp_path)
    monkeypatch.setattr(train_module, "_git_metadata", lambda: (COMMIT, True))
    monkeypatch.setattr(sweep, "train", lambda *args, **kwargs: pytest.fail("trained dirty repo"))
    with pytest.raises(ValueError, match="dirty"):
        _run(sweep, config_path, tmp_path)
    assert not (tmp_path / "results").exists()


def test_sweep_skips_existing_name_and_commit(tmp_path, monkeypatch):
    sweep = importlib.import_module("run_sweep")
    config_path = _tiny_config(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    existing = results_dir / "existing.json"
    existing.write_text(json.dumps({
        "config": {"name": "parity_sweep_test_s0"},
        "git_commit": COMMIT,
    }), encoding="utf-8")
    monkeypatch.setattr(train_module, "_git_metadata", lambda: (COMMIT, False))
    seen = []
    def fake_train(config, **kwargs):
        seen.append(config.seed)
        return SimpleNamespace(results_path=tmp_path / f"{config.name}.json")

    monkeypatch.setattr(sweep, "train", fake_train)
    old_threads = torch.get_num_threads()
    try:
        _run(sweep, config_path, tmp_path, seeds=(0, 1))
        assert seen == [1]
    finally:
        torch.set_num_threads(old_threads)


def test_sweep_runs_existing_name_when_commit_differs(tmp_path, monkeypatch):
    sweep = importlib.import_module("run_sweep")
    config_path = _tiny_config(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "old_commit.json").write_text(json.dumps({
        "config": {"name": "parity_sweep_test_s0"},
        "git_commit": "b" * 40,
    }), encoding="utf-8")
    monkeypatch.setattr(train_module, "_git_metadata", lambda: (COMMIT, False))
    seen = []

    def fake_train(config, **kwargs):
        seen.append(config.seed)
        return SimpleNamespace(results_path=tmp_path / f"{config.name}.json")

    monkeypatch.setattr(sweep, "train", fake_train)
    old_threads = torch.get_num_threads()
    try:
        _run(sweep, config_path, tmp_path, seeds=(0,))
        assert seen == [0]
    finally:
        torch.set_num_threads(old_threads)


def test_sweep_real_tiny_run_records_name_seed_and_threads(tmp_path, monkeypatch):
    sweep = importlib.import_module("run_sweep")
    config_path = _tiny_config(tmp_path)
    monkeypatch.setattr(train_module, "_git_metadata", lambda: (COMMIT, False))
    old_threads = torch.get_num_threads()
    try:
        _run(sweep, config_path, tmp_path, seeds=(7,), cpu_threads=2)
        files = list((tmp_path / "results").glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert files[0].name.endswith("_parity_sweep_test_s7.json")
        assert payload["config"]["name"] == "parity_sweep_test_s7"
        assert payload["config"]["seed"] == payload["seed"] == 7
        assert payload["cpu_threads"] == 2
        assert payload["git_commit"] == COMMIT
    finally:
        torch.set_num_threads(old_threads)
