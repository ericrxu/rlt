# RLT: an independent implementation of the Recurrent Looped Transformer

A from-scratch implementation of the **Recurrent Looped Transformer** (Zhang, Feng & Qin, 2026), built from the technical report, verified against its formal properties, and used to study length generalization on state-tracking tasks.

- **Report:** [`REPORT.md`](REPORT.md), the six experiments, figures, and discussion
- **Lab notebook:** [`RESULTS.md`](RESULTS.md), every run, per-seed results, and caveats
- **Design decisions and invariants:** [`AGENTS.md`](AGENTS.md)
- **Original report and project page:** [yifanzhang-pro.github.io/recurrent-looped-tranformer](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/)

## What is RLT?

RLT combines a causal transformer encoder with a decoder that processes tokens sequentially. At each position, a gated merge combines the encoder representation with the previous decoder output. The decoder then applies sliding-window attention to its recent history, cross-attention to encoder memory restricted to the current prefix, and a feed-forward block, in that order at every layer.

Both the decoder output and its per-layer attention caches carry forward to the next token. The output feeds into the next gated merge; the caches supply recent keys and values for sliding-window attention. In this repository, **feedback** refers specifically to the output-to-merge path: removing it leaves the attention caches and encoder memory intact. See [Section 2 of the report](REPORT.md#2-implementation-and-verification) for implementation and verification details.

## Findings

With final-answer-only supervision, RLT's length generalization depends on the training schedule. Under a short-to-long curriculum, removing its hidden-state feedback has little effect on short-program accuracy before the model collapses at longer lengths. With mixed or interleaved lengths, RLT generalizes perfectly or nearly so on every length evaluated, up to 32× the maximum training length, and removing its feedback drops accuracy to chance. These ablations support an attention-shortcut explanation for the curriculum failure; they do not establish its cause.

Final-state accuracy at length 128, mean over 3 seeds, **parity / five-state** (chance 50% / 20%):

| Experiment | Supervision | Training lengths | RLT | GRU |
| --- | --- | --- | --- | --- |
| 1 | Every position | Fixed 32 | 95.6% / 100% | 100% / 100% |
| 2 | Final only | Fixed 32 | 50.1% / 19.7% | 50.0% / 20.1% |
| 3 | Final only | Curriculum 4 → 32 | 49.9% / 19.8% | 100% / 100% |
| 5 | Final only | Mixed 4–32 in every step | 100% / 100% | 100% / 100% |
| 6 | Final only | Interleaved 4–32 | 99.98% / 100% | 100% / 100% |

![Length generalization](results/figures/length_generalization.png)

Thick lines are means over 3 seeds; thin lines are individual seeds. Maximum training length 32 (dashed line). See the report for the full figure caption and the diagnosis of Experiment 3.

All models have about 59K parameters, are matched in parameter count but not compute, and are trained on two synthetic tasks. These results say nothing yet about RLT at scale.

## What's here

- **The model:** causal encoder, prefix-restricted encoder memory, gated merge, and a recurrent decoder with per-layer sliding-window caches and a learned initial state, trained with full backpropagation through time.
- **Baselines:** RLT with α = 0 (feedback off), a decoder-only transformer, and a GRU, parameter-matched within 5%.
- **Verification:** 392 tests, including the report's serving-split invariance (Proposition 3.1, all split points, to `1e-12`), full-state causality (Proposition B.1), the two-gradient-path analysis, and finite-difference gradient checks.
- **Reproducibility:** training is deterministic on one CPU thread, and reruns reproduce earlier loss histories bit for bit across commits.

```
model.py, encoder.py, decoder.py,      RLT, component by component
memory.py, merge.py, window_cache.py,
layers.py
baselines.py                           Transformer and GRU baselines
task.py                                Parity and five-state program generators
objectives.py                          Losses and accuracy metrics
train.py                               Deterministic training runner
run_sweep.py                           Runs one config over several seeds
compare.py                             Aggregates results; refuses unfair comparisons
configs/                               One JSON file per model, task, and experiment
results/                               Results for every run, plus figures
analysis/                              Diagnostic and plotting scripts
tests/                                 The test suite
```

## Setup

Requires Python 3.10 or later. Everything runs on CPU.

```
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash); .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Reproducing an experiment

Each experiment is a set of configs run over seeds 0, 1, and 2. For example, to reproduce one condition of Experiment 5:

```
python run_sweep.py --config configs/parity_rlt_mixed_final.json --cpu-threads 1 --results-dir results/exp5 --checkpoint-dir checkpoints/exp5
python compare.py results/exp5
```

Repeat the sweep for each configuration below to reproduce the full experiment. Config names have the `.json` extension; `compare.py` summarizes the conditions present in the results directory.

| Experiment | Configs (for `{task}` = `parity`, `five_state`) |
| --- | --- |
| 1 | `{task}_rlt`, `{task}_rlt_alpha0`, `{task}_transformer`, `{task}_gru` |
| 2 | `{task}_rlt_final`, `{task}_gru_final` |
| 3 | `{task}_rlt_curriculum`, `{task}_gru_curriculum` |
| 4 | `{task}_rlt_a05_curriculum`, `{task}_rlt_a01_curriculum` |
| 5 | `{task}_rlt_mixed_final`, `{task}_gru_mixed_final` |
| 6 | `{task}_rlt_cycle_final`, `{task}_gru_cycle_final` |

- Use `--cpu-threads 1` to reproduce results bit for bit.
- `run_sweep.py` refuses to run on uncommitted changes; every results file records its commit.
- Checkpoints are not committed. The scripts in `analysis/` need them, so rerun the relevant experiment first.
- An RLT run takes 6–25 minutes on a laptop CPU; baseline runs take a few minutes.

### Reproducing the report's analyses

Run these commands from the repository root after training the relevant conditions. The extended evaluations use final checkpoints; the feedback ablation also needs Experiment 4's stage checkpoints for both tasks and both α values, plus the final RLT checkpoints from Experiments 5 and 6.

```sh
python -u analysis/eval_long.py results/exp1 checkpoints/exp1 "*_rlt_s*" > results/long_eval/exp1_rlt.txt
python -u analysis/eval_long.py results/exp1 checkpoints/exp1 "*_gru_s*" > results/long_eval/exp1_gru.txt
python -u analysis/eval_long.py results/exp5 checkpoints/exp5 > results/long_eval/exp5.txt
python -u analysis/eval_long.py results/exp6 checkpoints/exp6 > results/long_eval/exp6.txt
python analysis/feedback_ablation.py
python analysis/plot_lengths.py
python analysis/plot_collapse.py
python analysis/plot_ablation.py
python analysis/plot_schedule_losses.py
```

The plotting scripts write the four report figures to `results/figures/`; the collapse plot reads Experiment 3's training results, and the schedule comparison reads RLT loss histories from Experiments 3, 5, and 6. Extended evaluation uses 1,024 programs per length, and the feedback ablation uses 512 per condition.

## Departures from the report

- **Objective:** training uses the tasks' state labels rather than the report's next-token objective, since the next input token in these tasks is random.
- **Scale:** width 32, 2 encoder and 2 decoder layers, window 3. The authors' experiments use width 512 and 8 or 16 layers.
- Other choices the report leaves open are listed in [`AGENTS.md`](AGENTS.md).

## How this was built

Test-first, with an AI coding assistant: for each component, tests were written and locked before any implementation, and a separate session implemented against them without modifying the tests.

## Citation

```
@techreport{zhang2026recurrentlooped,
  title  = {Recurrent Looped Transformer},
  author = {Zhang, Yifan and Feng, Jichen and Qin, Shihan},
  year   = {2026},
  month  = sep,
  url    = {https://github.com/yifanzhang-pro/recurrent-looped-tranformer}
}
```
