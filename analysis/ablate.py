import glob, os, sys
sys.path.insert(0, os.getcwd())
import torch
from train import TrainConfig, build_model, evaluate

path = glob.glob(os.path.expanduser("~/pilot_results/*short_rlt.pt"))[0]
checkpoint = torch.load(path, weights_only=False)
for alpha in (0.5, 0.0):
    config = TrainConfig(**({"curriculum": None} | checkpoint["config"] | {"alpha": alpha}))
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    print(f"alpha={alpha}:", evaluate(model, config))