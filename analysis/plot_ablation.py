import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

records = json.load(open("results/ablations/feedback_cut.json", encoding="utf-8"))
groups = list(dict.fromkeys(record["group"] for record in records))
width = 0.36

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, task, title, chance in zip(axes, ("parity", "five_state"),
                                   ("Parity", "Five-state"), (0.5, 0.2)):
    tick_labels = []
    for index, group in enumerate(groups):
        runs = [r for r in records if r["task"] == task and r["group"] == group]
        tick_labels.append(f"{group}\nlength {runs[0]['length']}")
        for offset, key, color, label in (
                (-width / 2, "feedback_on", "tab:blue", "Feedback on"),
                (width / 2, "feedback_cut", "tab:red", "Feedback cut (α = 0 at test)")):
            values = [run[key] for run in runs]
            ax.bar(index + offset, sum(values) / len(values), width, color=color,
                   alpha=0.75, label=label if index == 0 else None)
            ax.scatter([index + offset] * len(values), values, color="black", s=14,
                       zorder=3,
                       label="Individual seeds" if index == 0 and key == "feedback_on" else None)
    ax.axhline(chance, color="red", linestyle=":", linewidth=1, label="Chance")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
axes[0].set_ylabel("Final-state accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig("results/figures/feedback_ablation.png", dpi=150, bbox_inches="tight")
print("saved results/figures/feedback_ablation.png")