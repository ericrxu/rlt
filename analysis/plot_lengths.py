import glob, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = [  # label, results folder, run name, long-eval file, color
    ("Transformer (Exp 1)", "results/exp1", "{task}_transformer", None, "tab:gray"),
    ("RLT, α = 0 (Exp 1)", "results/exp1", "{task}_rlt_alpha0", None, "tab:brown"),
    ("RLT (Exp 1, per-position)", "results/exp1", "{task}_rlt",
     "results/long_eval/exp1_rlt.txt", "tab:orange"),
    ("RLT (Exp 6, interleaved)", "results/exp6", "{task}_rlt_cycle_final",
     "results/long_eval/exp6.txt", "tab:green"),
    ("RLT (Exp 5, mixed)", "results/exp5", "{task}_rlt_mixed_final",
     "results/long_eval/exp5.txt", "tab:blue"),
    ("GRU (Exp 5, mixed)", "results/exp5", "{task}_gru_mixed_final",
     "results/long_eval/exp5.txt", "black"),
]
RUN_LINE = re.compile(r"^(\S+_s\d+)\s+(.*)$")
SCORE = re.compile(r"(\d+): ([0-9.]+)")


def long_scores(path):
    scores = {}
    if path is None:
        return scores
    for line in open(path, encoding="utf-8"):
        match = RUN_LINE.match(line.strip())
        if match:
            scores[match.group(1)] = {int(length): float(value)
                                      for length, value in SCORE.findall(match.group(2))}
    return scores


def seed_curves(results_dir, run_name, long_path):
    extra = long_scores(long_path)
    curves = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        run = json.load(open(path, encoding="utf-8"))
        name = run["config"]["name"]
        if name.rsplit("_s", 1)[0] != run_name:
            continue
        points = {int(length): metrics["final_state_accuracy"]
                  for length, metrics in run["eval_history"][-1]["lengths"].items()}
        for length, value in extra.get(name, {}).items():
            if length > max(points):
                points[length] = value
        curves[run["seed"]] = dict(sorted(points.items()))
    return curves


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, task, title, chance in zip(axes, ("parity", "five_state"),
                                   ("Parity", "Five-state"), (0.5, 0.2)):
    for label, results_dir, run_name, long_path, color in SERIES:
        curves = seed_curves(results_dir, run_name.format(task=task), long_path)
        if not curves:
            print("no runs found for", label, task)
            continue
        for curve in curves.values():
            ax.plot(list(curve), list(curve.values()), color=color, alpha=0.25, linewidth=1)
        lengths = sorted(set.intersection(*(set(curve) for curve in curves.values())))
        means = [sum(curve[length] for curve in curves.values()) / len(curves)
                 for length in lengths]
        style = "--" if label.startswith("GRU") else "-"
        ax.plot(lengths, means, color=color, linewidth=2.2, marker="o", markersize=3.5,
                linestyle=style, label=label)
    ax.axhline(chance, color="red", linestyle=":", linewidth=1, label="Chance")
    ax.axvline(32, color="gray", linestyle="--", linewidth=1, label="Training length")
    ax.set_xscale("log", base=2)
    ticks = [32, 64, 128, 256, 512, 1024]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Program length (log scale)")
    ax.set_title(title)
axes[0].set_ylabel("Final-state accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/length_generalization.png", dpi=150, bbox_inches="tight")
print("saved results/figures/length_generalization.png")