import glob, os, sys
from dataclasses import replace
sys.path.insert(0, os.getcwd())
import torch
from train import TrainConfig, build_model, evaluate

results_dir, checkpoint_dir = sys.argv[1], sys.argv[2]
for path in sorted(glob.glob(os.path.join(results_dir, "*rlt*.json"))):
    stem = os.path.basename(path)[:-5]
    checkpoint = torch.load(os.path.join(checkpoint_dir, f"{stem}.pt"), weights_only=True)
    base = TrainConfig(**checkpoint["config"])
    scores = []
    for alpha in (base.alpha, 0.0):
        config = replace(base, alpha=alpha, eval_lengths=(32, 128), eval_programs=512)
        model = build_model(config)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(model, config)
        scores.append(f"{metrics['32']['final_state_accuracy']:.3f} / "
                      f"{metrics['128']['final_state_accuracy']:.3f}")
    print(f"{stem.split('_', 1)[1]}: feedback on {scores[0]}   cut {scores[1]}"
          "   (len 32 / len 128)")