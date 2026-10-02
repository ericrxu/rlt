import glob, json, os, sys
from statistics import mean

stages = [(1, 1500, 4), (1501, 3000, 8), (3001, 4500, 16), (4501, 6000, 32)]
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*rlt*.json"))):
    run = json.load(open(path))
    print(os.path.basename(path).split("_", 1)[1])
    losses, diags = run["loss_history"], run["diagnostics"]
    for start, end, length in stages:
        seg = losses[start - 1:end]
        dseg = diags[start - 1:end]
        solved = next((start + i for i, v in enumerate(seg) if v < 0.01), None)
        print(f"  len {length:>2}: end loss {mean(seg[-100:]):.3f}"
              f"  first<0.01 at step {solved}"
              f"  max grad {max(d['preclip_grad_norm'] for d in dseg):.1f}"
              f"  state norm {mean(d['mean_state_norm'] for d in dseg[-100:]):.1f}"
              f"  gate {mean(d['mean_gate'] for d in dseg[-100:]):.3f}")