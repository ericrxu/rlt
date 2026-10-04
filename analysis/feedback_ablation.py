import glob, json, os, sys
from dataclasses import replace
sys.path.insert(0, os.getcwd())
import torch
from train import TrainConfig, build_model, evaluate

BEFORE_COLLAPSE = {"parity": (1500, 4), "five_state": (3000, 8)}
FINAL = {"parity": (None, 32), "five_state": (None, 32)}
DEFAULTS = {"objective": "per_position", "curriculum": None,
            "length_mix": None, "length_cycle": None}
GROUPS = [  # label, results folder, checkpoint folder, run name, (step, length) by task
    ("Curriculum, α = 0.5\n(before collapse)", "results/exp4", "checkpoints/exp4",
     "{task}_rlt_a05_curriculum", BEFORE_COLLAPSE),
    ("Curriculum, α = 0.1\n(before collapse)", "results/exp4", "checkpoints/exp4",
     "{task}_rlt_a01_curriculum", BEFORE_COLLAPSE),
    ("Mixed lengths\n(Exp 5, final)", "results/exp5", "checkpoints/exp5",
     "{task}_rlt_mixed_final", FINAL),
    ("Interleaved\n(Exp 6, final)", "results/exp6", "checkpoints/exp6",
     "{task}_rlt_cycle_final", FINAL),
]

records = []
for label, results_dir, checkpoint_dir, run_name, settings in GROUPS:
    for task in ("parity", "five_state"):
        step, length = settings[task]
        for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
            run = json.load(open(path, encoding="utf-8"))
            if run["config"]["name"].rsplit("_s", 1)[0] != run_name.format(task=task):
                continue
            stem = os.path.basename(path)[:-5]
            suffix = f"_step{step}.pt" if step else ".pt"
            checkpoint = torch.load(os.path.join(checkpoint_dir, stem + suffix),
                                    weights_only=True)
            base = TrainConfig(**(DEFAULTS | checkpoint["config"]))
            scores = {}
            for key, alpha in (("feedback_on", base.alpha), ("feedback_cut", 0.0)):
                config = replace(base, alpha=alpha, eval_lengths=(length,),
                                 eval_programs=512)
                model = build_model(config)
                model.load_state_dict(checkpoint["model"])
                scores[key] = evaluate(model, config)[str(length)]["final_state_accuracy"]
            records.append({"group": label, "task": task, "seed": run["seed"],
                            "alpha": base.alpha, "length": length, **scores})
            print(label.replace("\n", " "), task, "seed", run["seed"],
                  f"on {scores['feedback_on']:.3f}  cut {scores['feedback_cut']:.3f}",
                  flush=True)

os.makedirs("results/ablations", exist_ok=True)
with open("results/ablations/feedback_cut.json", "w", encoding="utf-8") as file:
    json.dump(records, file, indent=2)
print("saved results/ablations/feedback_cut.json")