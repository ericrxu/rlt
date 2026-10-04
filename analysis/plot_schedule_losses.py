"""Compare saved RLT training losses under three final-only length schedules.

Run from the repository root: python analysis/plot_schedule_losses.py
"""

import json
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
WINDOW = 24  # Six complete four-length cycles once the window is full.
SCHEDULES = (
    ("Curriculum (Exp 3)", "exp3", "curriculum", "tab:orange", 6000),
    ("Mixed (Exp 5)", "exp5", "mixed_final", "tab:blue", 3000),
    ("Interleaved (Exp 6)", "exp6", "cycle_final", "tab:green", 3000),
)


def moving_average(values):
    total = 0.0
    averaged = []
    for index, value in enumerate(values):
        total += value
        if index >= WINDOW:
            total -= values[index - WINDOW]
        averaged.append(total / min(index + 1, WINDOW))
    return averaged


def load_curves(task, folder, suffix, steps):
    curves = {}
    for path in sorted((ROOT / "results" / folder).glob(
            f"*_{task}_rlt_{suffix}_s*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        config = run["config"]
        seed = run["seed"]
        if run["dirty"]:
            raise ValueError(f"Dirty training run: {path}")
        if seed in curves:
            raise ValueError(f"Duplicate seed {seed}: {path}")
        if (config["task"] != task or config["model_type"] != "rlt"
                or config["objective"] != "final_state"
                or config["alpha"] != 0.5 or config["steps"] != steps
                or config["seed"] != seed):
            raise ValueError(f"Unexpected configuration: {path}")
        losses = run["loss_history"]
        if len(losses) != steps or not all(
                math.isfinite(value) and value >= 0 for value in losses):
            raise ValueError(f"Invalid loss history: {path}")
        curves[seed] = moving_average(losses)
    if set(curves) != {0, 1, 2}:
        raise ValueError(f"Expected seeds 0, 1, 2 for {task}, {folder}")
    return [curves[seed] for seed in sorted(curves)]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for ax, task, title, classes in zip(
            axes, ("parity", "five_state"), ("Parity", "Five-state"), (2, 5)):
        for label, folder, suffix, color, steps in SCHEDULES:
            curves = load_curves(task, folder, suffix, steps)
            positions = range(1, steps + 1)
            for curve in curves:
                ax.plot(positions, curve, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(positions, [mean(values) for values in zip(*curves)],
                    color=color, linewidth=2, label=label)
        ax.axhline(math.log(classes), color="0.4", linestyle=":", linewidth=1)
        for boundary in (1500, 3000, 4500):
            ax.axvline(boundary, color="0.65", linestyle="--", linewidth=0.8,
                       zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.set_xlim(0, 6000)
        ax.set_ylim(bottom=0)
        ax.set_xticks(range(0, 6001, 1500))
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.98, 0.97, "Mixed / interleaved end at 3,000",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    axes[0].set_ylabel("Final-state training loss\n(24-step trailing average)")
    handles, _ = axes[0].get_legend_handles_labels()
    handles.extend((
        Line2D([], [], color="0.4", linestyle=":", label="Uniform-prediction loss"),
        Line2D([], [], color="0.65", linestyle="--", label="Curriculum transitions"),
    ))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    output = ROOT / "results" / "figures" / "schedule_losses.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
