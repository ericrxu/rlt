import glob, os, sys
from dataclasses import replace
from statistics import mean
sys.path.insert(0, os.getcwd())
import torch
from train import TrainConfig, build_model, evaluate

DEFAULTS = {"objective": "per_position", "curriculum": None, "length_mix": None,
            "length_cycle": None}
LENGTHS = (128, 256, 512, 1024)

results_dir, checkpoint_dir = sys.argv[1], sys.argv[2]
pattern = sys.argv[3] if len(sys.argv) > 3 else "*"
table = {}
for path in sorted(glob.glob(os.path.join(results_dir, f"{pattern}.json"))):
    stem = os.path.basename(path)[:-5]
    checkpoint_path = os.path.join(checkpoint_dir, f"{stem}.pt")
    if not os.path.exists(checkpoint_path):
        print("missing checkpoint:", checkpoint_path)
        continue
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    config = TrainConfig(**(DEFAULTS | checkpoint["config"]))
    config = replace(config, eval_lengths=LENGTHS, eval_programs=1024, batch_size=256)
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(model, config)
    scores = [metrics[str(length)]["final_state_accuracy"] for length in LENGTHS]
    run = stem.split("_", 1)[1]
    table.setdefault(run.rsplit("_s", 1)[0], []).append(scores)
    print(run, "  ".join(f"{length}: {score:.3f}" for length, score in zip(LENGTHS, scores)),
          flush=True)

print("\nMean over seeds")
print(f"{'':32s}" + "".join(f"{length:>8d}" for length in LENGTHS))
for name, rows in table.items():
    print(f"{name:32s}" + "".join(f"{mean(column):8.3f}" for column in zip(*rows)))