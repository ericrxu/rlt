"""Summarize comparable experiment results across seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


TRAINING_FIELDS = (
    "train_length", "batch_size", "train_programs", "steps", "learning_rate",
    "min_learning_rate", "warmup_steps", "weight_decay", "grad_clip_norm",
)
EVAL_FIELDS = ("eval_seed", "eval_lengths", "eval_programs", "eval_interval")


def compare_results(
    results_dir: str | Path, expected_seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict]:
    """Report final state accuracy while rejecting incomplete or unfair sets."""
    paths = sorted(Path(results_dir).glob("*.json"))
    if not paths:
        raise ValueError("no result files found")

    groups = defaultdict(dict)
    commit = None
    training_settings = None
    eval_settings = None
    objective = None
    unset_curriculum = object()
    curriculum = unset_curriculum
    unset_length_mix = object()
    length_mix = unset_length_mix
    unset_length_cycle = object()
    length_cycle = unset_length_cycle
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        if payload["dirty"]:
            raise ValueError(f"dirty result: {path}")
        if commit is None:
            commit = payload["git_commit"]
        elif payload["git_commit"] != commit:
            raise ValueError("results have different commits")

        training = tuple(config[field] for field in TRAINING_FIELDS)
        evaluation = tuple(
            tuple(config[field]) if field == "eval_lengths" else config[field]
            for field in EVAL_FIELDS
        )
        if training_settings is None:
            training_settings = training
            eval_settings = evaluation
        if training != training_settings:
            raise ValueError("training settings differ")
        if evaluation != eval_settings:
            raise ValueError("eval settings differ")
        result_objective = config.get("objective", "per_position")
        if objective is None:
            objective = result_objective
        elif result_objective != objective:
            raise ValueError("results have different objectives")
        result_curriculum = config.get("curriculum")
        if curriculum is unset_curriculum:
            curriculum = result_curriculum
        elif result_curriculum != curriculum:
            raise ValueError("results have different curriculum settings")
        result_length_mix = config.get("length_mix")
        if length_mix is unset_length_mix:
            length_mix = result_length_mix
        elif result_length_mix != length_mix:
            raise ValueError("results have different length_mix settings")
        result_length_cycle = config.get("length_cycle")
        if length_cycle is unset_length_cycle:
            length_cycle = result_length_cycle
        elif result_length_cycle != length_cycle:
            raise ValueError("results have different length_cycle settings")

        key = (config["task"], config["model_type"], config["alpha"])
        seed = payload["seed"]
        if seed != config["seed"]:
            raise ValueError(f"seed mismatch in {path}")
        if seed in groups[key]:
            raise ValueError(f"duplicate seed {seed} in {key}")
        groups[key][seed] = payload

    expected = set(expected_seeds)
    summary = []
    for (task, model_type, alpha), runs in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))
    ):
        if set(runs) != expected:
            raise ValueError(f"missing or unexpected seed for {(task, model_type, alpha)}")
        lengths = {}
        for length in eval_settings[1]:
            scores = [
                runs[seed]["eval_history"][-1]["lengths"][str(length)]["final_state_accuracy"]
                for seed in sorted(expected)
            ]
            lengths[str(length)] = {
                "mean": mean(scores), "min": min(scores), "max": max(scores),
            }
        summary.append({
            "task": task, "model_type": model_type, "alpha": alpha,
            "lengths": lengths,
        })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    print(json.dumps(compare_results(args.results_dir, tuple(args.seeds)), indent=2))
