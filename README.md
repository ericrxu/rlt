# RLT: a from-scratch implementation of the Recurrent Looped Transformer

An independent implementation of the **Recurrent Looped Transformer** (Zhang, Feng & Qin, technical report, September 2026), built from the paper alone, verified against its formal properties, and used to run a series of controlled length-generalization experiments.

- **Paper and project page:** [github.com/yifanzhang-pro/recurrent-looped-tranformer](https://github.com/yifanzhang-pro/recurrent-looped-tranformer)
- **Full experimental record:** [`RESULTS.md`](RESULTS.md)
- **Design decisions and invariants:** [`AGENTS.md`](AGENTS.md)

## Headline finding

RLT's recurrence generalizes far beyond the training length once it is learned, but **whether it gets learned depends on the training data's length schedule**. When RLT trains for long stretches on short programs only, it solves them through attention instead of recurrence, and later collapses. When long programs appear frequently, it learns its recurrence and holds 100% accuracy to 32× the training length.

Final-state accuracy at length 128 (4× the training length), mean over 3 seeds, **parity / five-state**:

| Experiment | Supervision | Training lengths | RLT | GRU |
| --- | --- | --- | --- | --- |
| 1 | Every position | Fixed 32 | 95.6% / 100% | 100% / 100% |
| 2 | Final answer only | Fixed 32 | 50.1% / 19.7% | 50.0% / 20.1% |
| 3 | Final answer only | Curriculum 4 → 8 → 16 → 32 | 49.9% / 19.8% | 100% / 100% |
| 5 | Final answer only | Mixed 4–32 in every step | **100% / 100%** | 100% / 100% |
| 6 | Final answer only | Interleaved 4–32, one per step | **99.98% / 100%** | 100% / 100% |

Chance is 50% for parity and 20% for five-state. In Experiment 1, RLT with its feedback removed (α = 0) and a parameter-matched transformer both score at chance at length 128. Experiment 4 diagnosed Experiment 3 and is described below.

### Accuracy by length

Final-state accuracy, mean over 3 seeds. Models are trained at length 32.

**Parity** (chance 50%)

| Model | 32 | 64 | 128 | 256 | 512 | 1,024 |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer (Exp 1) | 98.5% | 49.9% | 49.5% | — | — | — |
| RLT, α = 0 (Exp 1) | 90.7% | 51.5% | 50.0% | — | — | — |
| RLT (Exp 1) | 100% | 99.95% | 95.6% | 87.5% | 83.2% | 83.7% |
| RLT (Exp 6, interleaved) | 100% | 100% | 99.98% | 99.5% | 98.5% | 97.6% |
| RLT (Exp 5, mixed) | 100% | 100% | 100% | 100% | 100% | 100% |
| GRU (Exp 1, 5, 6) | 100% | 100% | 100% | 100% | 100% | 100% |

**Five-state** (chance 20%)

| Model | 32 | 64 | 128 | 256 | 512 | 1,024 |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer (Exp 1) | 22.2% | 20.0% | 20.7% | — | — | — |
| RLT, α = 0 (Exp 1) | 22.8% | 19.1% | 20.8% | — | — | — |
| RLT (Exp 1, 5, 6) | 100% | 100% | 100% | 100% | 100% | 99.9–100% |
| GRU (Exp 1, 5, 6) | 100% | 100% | 100% | 100% | 100% | 99.9–100% |

Lengths 32–128 use 2,048 test programs per length; 256–1,024 use 1,024 test programs, evaluated afterward from saved checkpoints. Dashes mark models not evaluated past 128, since they are already at chance. Experiment 1's parity RLT average hides a split: two seeds hold 100% to 1,024, while the third falls to chance by 512 (see `RESULTS.md`).

All models here have about 59K parameters and are trained on two synthetic tasks. These results say nothing yet about RLT at scale.

## What was built

The model follows the paper's specification:

- **Causal encoder** (token-parallel), with RoPE at explicit absolute positions.
- **Encoder-derived KV memory**, prefix-restricted so the decoder never sees future memory. One memory group, or one per decoder layer.
- **Gated merge** of the current token's encoder output with the previous decoder state: `u_t = e_t + α · g_t ⊙ W_s · RMSNorm(s_{t-1})`.
- **Recurrent decoder** with per-layer sliding-window attention caches, cross-attention to memory, and a learned initial state `s*`. The full state is the pair `(s_t, C_t)`, carried across every token.
- **Full backpropagation through time**, with the decoder loop strictly sequential.

Baselines, parameter-matched within 5%: RLT with α = 0 (same model, feedback off), a decoder-only transformer, and a GRU.

## Verification

Every component was built test-first. **392 tests** pass, including checks of the paper's formal properties:

- **Proposition 3.1 (serving-split invariance):** prefilling a prompt and then stepping token by token gives identical states and predictions at every split point, verified at all 11 splits across 8 configurations to `1e-12`.
- **Proposition B.1 (causality):** changing token *j* leaves the full state at every earlier position bit-identical.
- **Two gradient paths (Appendices B–C):** with window size above 1, gradient reaches the initial state through both the recurrent state and the decoder's window cache; detaching each path separately, and both together, behaves as the paper describes.
- **Finite differences:** autograd gradients match central differences to `rtol = 1e-6`.

Training is deterministic on one CPU thread. Reruns reproduce earlier loss histories **bit for bit across commits**: Experiment 4's α = 0.5 arm reproduces Experiment 3's 6,000-step histories exactly, and Experiment 1 was re-verified after each change to the training loop.

## The experiments

Full details, per-seed results, and caveats for each are in [`RESULTS.md`](RESULTS.md).

1. **RLT vs. baselines (per-position labels).** The recurrence is decisive: RLT generalizes to 4× training length, while α = 0 and the transformer are at chance. The transformer never learns five-state (an S5 word problem) even at the training length.
2. **Final-answer-only supervision.** Both RLT and the GRU fail to learn at all at this budget. A positive control at length 4 confirms the training path works.
3. **Curriculum.** The GRU learns perfectly; RLT learns each short stage and then collapses at the length transitions.
4. **Diagnosis.** Stage-end checkpoints show that right before each collapse, cutting RLT's feedback barely changes its accuracy: it had solved short programs through attention, not recurrence. The collapse is to a single constant answer. A weaker feedback scale (α = 0.1) collapses identically.
5. **Mixed lengths.** Predicted by the diagnosis: with all lengths in every step, RLT learns from final answers alone. Cutting its feedback now drops it to chance, so it relies entirely on recurrence.
6. **Interleaved lengths.** A control separating ordering from within-step mixing: one length per step, cycling, is enough to prevent the collapse.

**Long-length evaluation:** Experiment 5's models hold 100% to length 1,024 (32× training), with one five-state seed at 99.9%.

## Repository layout

```
model.py, encoder.py, decoder.py,      RLT, built component by component
memory.py, merge.py, window_cache.py,
layers.py
baselines.py                           Transformer and GRU baselines
task.py                                Parity and five-state program generators
objectives.py                          Losses and accuracy metrics
train.py                               Deterministic training runner
run_sweep.py                           Runs one config over several seeds
compare.py                             Aggregates results; refuses unfair comparisons
configs/                               One JSON file per model, task, and experiment
results/                               Results files for every run (committed)
analysis/                              Diagnostic and evaluation scripts
tests/                                 The test suite
AGENTS.md                              Project choices and invariants
RESULTS.md                             Lab notebook
```

## Setup

Requires Python 3.10 or later.

```
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash); use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m pytest tests/ -q
```

Everything runs on CPU.

## Reproducing an experiment

Each experiment is a set of configs run over seeds 0, 1, and 2. For example, Experiment 5:

```
python run_sweep.py --config configs/parity_rlt_mixed_final.json --cpu-threads 1 --results-dir results/exp5 --checkpoint-dir checkpoints/exp5
python run_sweep.py --config configs/parity_gru_mixed_final.json --cpu-threads 1 --results-dir results/exp5 --checkpoint-dir checkpoints/exp5
python run_sweep.py --config configs/five_state_rlt_mixed_final.json --cpu-threads 1 --results-dir results/exp5 --checkpoint-dir checkpoints/exp5
python run_sweep.py --config configs/five_state_gru_mixed_final.json --cpu-threads 1 --results-dir results/exp5 --checkpoint-dir checkpoints/exp5
python compare.py results/exp5
```

Configs by experiment, for `{task}` in `parity` and `five_state`:

| Experiment | Configs |
| --- | --- |
| 1 | `{task}_rlt`, `{task}_rlt_alpha0`, `{task}_transformer`, `{task}_gru` |
| 2 | `{task}_rlt_final`, `{task}_gru_final` |
| 3 | `{task}_rlt_curriculum`, `{task}_gru_curriculum` |
| 4 | `{task}_rlt_a05_curriculum`, `{task}_rlt_a01_curriculum` |
| 5 | `{task}_rlt_mixed_final`, `{task}_gru_mixed_final` |
| 6 | `{task}_rlt_cycle_final`, `{task}_gru_cycle_final` |

Notes:

- **Use `--cpu-threads 1`** to reproduce results bit for bit. Different thread counts can change results in the last digits.
- **Start from a clean checkout.** `run_sweep.py` refuses to run if tracked files have uncommitted changes, and every results file records its commit and whether the repository was clean.
- **Checkpoints are not committed.** The analysis scripts in `analysis/` need them, so rerun the relevant experiment first.
- Rough timings on a laptop CPU: an RLT run takes 6–25 minutes depending on the experiment; GRU and transformer runs take a few minutes. Running several sweeps in parallel terminals works well.

## Departures from the paper

- **Training objective.** The experiments train on the task's state labels (after every operation, or the final one only), not the paper's next-token objective (eq. 5.1). In these tasks the next input token is random, so next-token loss cannot distinguish architectures.
- **Scale.** Width 32, 2 encoder and 2 decoder layers, window 3. The authors' own experiments use width 512 and 8 or 16 layers.
- Remaining implementation choices the paper leaves open are listed in [`AGENTS.md`](AGENTS.md).

## How this was built

Development was test-first with an AI coding assistant: for each component, tests were written and locked before any implementation, and a separate session implemented against them without modifying the tests. `AGENTS.md` records the invariants and decisions every session had to follow.

## Citation

If you use this work, please cite the original report:

```
@techreport{zhang2026recurrentlooped,
  title  = {Recurrent Looped Transformer},
  author = {Zhang, Yifan and Feng, Jichen and Qin, Shihan},
  year   = {2026},
  month  = sep,
  url    = {https://github.com/yifanzhang-pro/recurrent-looped-tranformer}
}
```
