import glob, os, sys
from dataclasses import replace
sys.path.insert(0, os.getcwd())
import torch
from train import TrainConfig, build_model, evaluate

results_dir, checkpoint_dir = sys.argv[1], sys.argv[2]
stages = [(1500, 4, 8), (3000, 8, 16), (4500, 16, 32), (6000, 32, 64)]
for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    stem = os.path.basename(path)[:-5]
    print(stem.split("_", 1)[1])
    for step, length, following in stages:
        checkpoint = torch.load(os.path.join(checkpoint_dir, f"{stem}_step{step}.pt"),
                                weights_only=True)
        base = TrainConfig(**checkpoint["config"])
        scores = []
        for alpha in (base.alpha, 0.0):
            config = replace(base, alpha=alpha, eval_lengths=(length, following),
                             eval_programs=512)
            model = build_model(config)
            model.load_state_dict(checkpoint["model"])
            metrics = evaluate(model, config)
            scores.append(f"{metrics[str(length)]['final_state_accuracy']:.3f} / "
                          f"{metrics[str(following)]['final_state_accuracy']:.3f}")
        print(f"  step {step}: feedback on {scores[0]}   cut {scores[1]}"
              f"   (len {length} / len {following})")