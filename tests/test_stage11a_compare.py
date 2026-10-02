"""Stage 11a comparison contracts using synthetic results only."""

from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path

import pytest

from train import load_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
COMMIT = "a" * 40
LENGTHS = (32, 48)


def _config(variant="rlt", task="parity", **changes):
    source = CONFIG_DIR / f"{task}_{variant if variant != 'rlt_alpha0' else 'rlt'}.json"
    config = load_config(source)
    if variant == "rlt_alpha0":
        config = replace(config, name=f"{task}_rlt_alpha0", alpha=0.0)
    return replace(config, **({"eval_lengths": LENGTHS} | changes))


def _result(directory, variant, seed, scores, *, task="parity", dirty=False,
            commit=COMMIT, config_changes=None, file_name=None):
    config = replace(_config(variant, task, **(config_changes or {})), seed=seed)
    payload = {
        "config": asdict(config),
        "git_commit": commit,
        "dirty": dirty,
        "seed": seed,
        "eval_history": [
            {"step": 1000, "lengths": {
                str(length): {"final_state_accuracy": 0.99} for length in LENGTHS
            }},
            {"step": config.steps, "lengths": {
                str(length): {"final_state_accuracy": score}
                for length, score in zip(config.eval_lengths, scores)
            }},
        ],
    }
    path = directory / (file_name or f"{task}_{variant}_s{seed}.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _compare(directory):
    compare = importlib.import_module("compare")
    return compare.compare_results(directory, expected_seeds=(0, 1, 2))


def _group(summary, task, model_type, alpha):
    return next(group for group in summary if
                group["task"] == task and group["model_type"] == model_type
                and group["alpha"] == alpha)


def test_compare_uses_final_eval_and_reports_mean_min_max_by_length(tmp_path):
    for seed, scores in enumerate(((0.2, 0.3), (0.4, 0.5), (0.6, 0.7))):
        _result(tmp_path, "rlt", seed, scores)
    summary = _compare(tmp_path)
    assert len(summary) == 1
    group = _group(summary, "parity", "rlt", 0.5)
    assert group["lengths"] == {
        "32": {"mean": pytest.approx(0.4), "min": 0.2, "max": 0.6},
        "48": {"mean": pytest.approx(0.5), "min": 0.3, "max": 0.7},
    }


def test_compare_keeps_rlt_alpha_values_in_separate_groups(tmp_path):
    for seed in (0, 1, 2):
        _result(tmp_path, "rlt", seed, (0.8, 0.7))
        _result(tmp_path, "rlt_alpha0", seed, (0.2, 0.1))
    summary = _compare(tmp_path)
    assert len(summary) == 2
    assert _group(summary, "parity", "rlt", 0.5)["lengths"]["32"]["mean"] == 0.8
    assert _group(summary, "parity", "rlt", 0.0)["lengths"]["32"]["mean"] == 0.2


def test_compare_accepts_fair_rlt_transformer_and_gru_results(tmp_path):
    for seed in (0, 1, 2):
        _result(tmp_path, "rlt", seed, (0.2 + seed * 0.1, 0.3))
        _result(tmp_path, "transformer", seed, (0.5, 0.6))
        _result(tmp_path, "gru", seed, (0.8, 0.9))
    summary = _compare(tmp_path)
    assert len(summary) == 3
    assert _group(summary, "parity", "rlt", 0.5)["lengths"]["32"]["mean"] == pytest.approx(0.3)
    assert _group(summary, "parity", "transformer", None)["lengths"]["32"] == {
        "mean": 0.5, "min": 0.5, "max": 0.5,
    }
    assert _group(summary, "parity", "gru", None)["lengths"]["48"] == {
        "mean": 0.9, "min": 0.9, "max": 0.9,
    }


@pytest.mark.parametrize("defect", ("dirty", "commit", "missing_seed", "eval_seed",
                                     "eval_lengths", "eval_programs", "training",
                                     "duplicate"))
def test_compare_refuses_incompatible_or_incomplete_results(tmp_path, defect):
    for seed in (0, 1, 2):
        _result(tmp_path, "rlt", seed, (0.3, 0.4),
                dirty=defect == "dirty" and seed == 1,
                commit="b" * 40 if defect == "commit" and seed == 1 else COMMIT)

    if defect == "missing_seed":
        (tmp_path / "parity_rlt_s1.json").unlink()
    elif defect == "duplicate":
        _result(tmp_path, "rlt", 1, (0.3, 0.4), file_name="duplicate.json")
    elif defect in ("eval_seed", "eval_lengths", "eval_programs", "training"):
        changes = {
            "eval_seed": {"eval_seed": 1001},
            "eval_lengths": {"eval_lengths": (32, 64)},
            "eval_programs": {"eval_programs": 1024},
            "training": {"batch_size": 32},
        }[defect]
        for seed in (0, 1, 2):
            _result(tmp_path, "transformer", seed, (0.5, 0.6),
                    config_changes=changes)

    reason = {
        "dirty": "dirty",
        "commit": "commit",
        "missing_seed": "seed",
        "eval_seed": "eval",
        "eval_lengths": "eval",
        "eval_programs": "eval",
        "training": "training",
        "duplicate": "duplicate",
    }[defect]
    with pytest.raises(ValueError, match=reason):
        _compare(tmp_path)
