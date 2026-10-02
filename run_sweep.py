"""Run a training config over a list of seeds on a clean checkout."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

import train as train_module
from train import load_config, train


def run_sweep(
    config_path: str | Path,
    seeds: tuple[int, ...] | list[int],
    cpu_threads: int,
    results_dir: str | Path = "results",
    checkpoint_dir: str | Path = "checkpoints",
) -> list[Path]:
    """Train missing seeds; an existing result counts only at the current commit."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    commit, dirty = train_module._git_metadata()
    if dirty:
        raise ValueError("refusing to run sweep on a dirty repository")

    base = load_config(config_path)
    results_dir = Path(results_dir)
    existing = set()
    if results_dir.exists():
        for path in results_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("git_commit") == commit:
                existing.add(payload["config"]["name"])

    torch.set_num_threads(cpu_threads)
    paths = []
    for seed in seeds:
        config = replace(base, seed=seed, name=f"{base.name}_s{seed}")
        if config.name in existing:
            continue
        outcome = train(config, output_dir=results_dir, checkpoint_dir=checkpoint_dir)
        paths.append(outcome.results_path)
        existing.add(config.name)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cpu-threads", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    for result_path in run_sweep(
        args.config, args.seeds, args.cpu_threads,
        results_dir=args.results_dir, checkpoint_dir=args.checkpoint_dir,
    ):
        print(result_path)
