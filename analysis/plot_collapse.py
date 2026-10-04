import glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pattern):
    return [json.load(open(path, encoding="utf-8")) for path in sorted(glob.glob(pattern))]


def moving_average(values, window=25):
    averaged, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        averaged.append(total / min(index + 1, window))
    return averaged


fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True)
for column, (task, title, chance) in enumerate(
        [("parity", "Parity", 0.693), ("five_state", "Five-state", 1.609)]):
    rlt_runs = load(f"results/exp3/*_{task}_rlt_curriculum_s*.json")
    gru_runs = load(f"results/exp3/*_{task}_gru_curriculum_s*.json")
    for index, run in enumerate(gru_runs):
        axes[0, column].plot(range(1, len(run["loss_history"]) + 1),
                             moving_average(run["loss_history"]), color="black",
                             alpha=0.5, linewidth=1, label="GRU" if index == 0 else None)
    for run in rlt_runs:
        steps = range(1, len(run["loss_history"]) + 1)
        line, = axes[0, column].plot(steps, moving_average(run["loss_history"]),
                                     linewidth=1.3, label=f"RLT seed {run['seed']}")
        color = line.get_color()
        axes[1, column].plot(steps, [d["preclip_grad_norm"] for d in run["diagnostics"]],
                             color=color, linewidth=0.6, alpha=0.7)
        axes[2, column].plot(steps, [d["mean_state_norm"] for d in run["diagnostics"]],
                             color=color, linewidth=1.3)
    axes[0, column].axhline(chance, color="red", linestyle=":", linewidth=1,
                            label="Chance-level loss")
    axes[1, column].axhline(1.0, color="purple", linestyle="-.", linewidth=1,
                            label="Clip limit (1.0)")
    for row in range(3):
        for boundary in (1500, 3000, 4500):
            axes[row, column].axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
    for start, length in zip((0, 1500, 3000, 4500), (4, 8, 16, 32)):
        axes[0, column].text(start + 750, 1.02, f"length {length}",
                             transform=axes[0, column].get_xaxis_transform(),
                             ha="center", va="bottom", fontsize=9)
    axes[0, column].set_title(title, pad=20)
    axes[1, column].set_yscale("log")
    axes[2, column].set_yscale("log")
    axes[2, column].set_xlabel("Training step")
axes[0, 0].set_ylabel("Training loss\n(25-step average)")
axes[1, 0].set_ylabel("Gradient norm\n(before clipping)")
axes[2, 0].set_ylabel("Decoder state norm")
handles, labels = axes[0, 0].get_legend_handles_labels()
clip_handles, clip_labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles + clip_handles, labels + clip_labels, loc="lower center",
           ncol=6, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()
fig.savefig("results/figures/curriculum_collapse.png", dpi=150, bbox_inches="tight")
print("saved results/figures/curriculum_collapse.png")