# Results

All experiments use per-position state labeling, not the paper's next-token objective (5.1).

| Date | Commit | Config | Seed | What was tested | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | dba0055 | parity_overfit | 0 | Overfit gate: 100 fixed programs, length 32 | Train 100%, eval@32 100% | Plateau at ln 2 until ~step 100, then sudden drop. Grad spike 9.7 at step 111, clipped. Gate stable ~0.545. |
| 2026-09-30 | ef3b093 | parity_overfit_gru | 0 | Overfit gate | Train 100%, eval@32 100% | Learned the rule. Plateau at ln 2 until ~step 105, then sudden drop. |
| 2026-09-30 | ef3b093 | parity_overfit_transformer | 0 | Overfit gate | Train 100%, eval@32 49% | Memorized, did not generalize. Per-position eval 60%. Slow, noisy convergence (~1,300 steps), grad spikes to 9.5. |

## Experiment 1 — RLT vs α=0, transformer, GRU

**Commit:** `COMMIT_HASH`
**Date:** 2026-10-02
**Results:** `results/exp1/` (24 runs, all clean)

### Setup

- Train at length 32, fresh programs every step, 3,000 steps, batch 64.
- AdamW, lr 0.003 → 0.0003 (cosine), 200 warmup steps, weight decay 0.01, gradient clip 1.0.
- Eval on the same 2,048 programs per length for every model (eval_seed 1000), at 32 / 48 / 64 / 96 / 128.
- Models, parameter-matched within 5% (~59K):
  - RLT: α = 0.5, W = 3, G = 1, 2 encoder + 2 decoder layers, dim 32
  - RLT α = 0: identical, with feedback switched off
  - Transformer: 3 layers, dim 40
  - GRU: 1 layer, dim 98
- Seeds 0, 1, 2. Single CPU thread per run.
- Metric: final-state accuracy, mean over 3 seeds (min–max where seeds disagree).

### Parity (chance = 50%)

| Model | 32 | 48 | 64 | 96 | 128 |
| --- | --- | --- | --- | --- | --- |
| RLT (α = 0.5) | 100% | 100% | 99.95% | 99.1% | 95.6% (86.9–100) |
| RLT (α = 0) | 90.7% | 51.1% | 51.5% | 49.3% | 50.0% |
| Transformer | 98.5% | 49.6% | 49.9% | 50.0% | 49.5% |
| GRU | 100% | 100% | 100% | 100% | 100% |

### Five-state (chance = 20%)

| Model | 32 | 48 | 64 | 96 | 128 |
| --- | --- | --- | --- | --- | --- |
| RLT (α = 0.5) | 100% | 100% | 100% | 100% | 100% |
| RLT (α = 0) | 22.8% | 20.0% | 19.1% | 20.2% | 20.8% |
| Transformer | 22.2% | 20.0% | 20.0% | 20.6% | 20.7% |
| GRU | 100% | 100% | 100% | 100% | 100% |

### Findings

- **Recurrence is the decisive factor.** RLT with feedback (α = 0.5) generalizes to 4× the training length; the same model with α = 0 is at chance beyond length 32 on parity, and at chance even at length 32 on five-state.
- **The transformer does not extrapolate.** It fits parity at length 32 (98.5%) but drops to chance at 48. It never learns five-state (an S5 word problem), consistent with known limits of fixed-depth transformers.
- **RLT matches the GRU but does not beat it.** Both are perfect on five-state. On parity, the GRU stays at 100% while RLT degrades gradually with length (95.6% at 128; one seed at 86.9%).

### Comparison to the published proof-of-concept

Note: the September 12 version of the report contained no experiments. The authors' project page has since added their own results (eight-layer encoder–decoder splits, six algorithmic tasks, three seeds). Their parity results agree with ours: RLT holds 100% at 256 bits for the 5+3 and 7+1 splits while a decoder-only Transformer is near chance. They report permutation tracking as still difficult (4+4 at 55.70 ± 25.78% on swaps-based tracking at length 512); our five-state RLT reaches 100% at length 128, under per-position supervision and a different task. The 60.8% / 20.7% figures below are from an independent community proof-of-concept listed on the same page.

The proof-of-concept reported RLT at 60.8% (parity) and 20.7% (five-state) at length 128, with GRU at 100% and 99.97%. Our GRU and transformer numbers match theirs closely; our RLT is substantially higher (95.6% and 100%).

This is unlikely to be a leak: the α = 0 control shares the encoder, memory, cross-attention, and SWA cache with full RLT, so any leak through those paths would also lift α = 0, which instead sits at chance. The full-state causality test (Prop. B.1) passes with feedback on. Likely differences: we supervise every position rather than only the final state; our five-state automaton (a transposition plus a 5-cycle) may differ from theirs; and their training budget, α, window size, and positional encoding are unknown.

### Caveats

- One setting per model (α = 0.5, W = 3, one learning rate); nothing tuned. The transformer might do better with its own settings.
- Compute is not matched; RLT is much slower per step than the GRU.
- Per-position state labeling, not the paper's next-token objective (5.1).
- Small models on synthetic tasks; says nothing about RLT at scale.

## Experiment 2 — final-state-only supervision

**Commit:** 69a692b9aad8f01e14888b69b5e3a15a51176fd2
**Date:** 2026-10-02
**Results:** `results/exp2/` (12 runs, all clean); positive control in `results/exp2_control/`

### Question

Does per-position supervision explain why our RLT generalizes far better than the community proof-of-concept?

### Setup

Identical to Experiment 1 except `objective: final_state`: the loss scores only the state after the last operation. Models: RLT (α = 0.5) and GRU, the GRU serving as a control. Seeds 0, 1, 2.

### Results: final-state accuracy, mean over 3 seeds

| Model | Task | 32 | 128 | Exp 1 at 128 (per-position) |
| --- | --- | --- | --- | --- |
| RLT | Parity | 49.8% | 50.1% | 95.6% |
| GRU | Parity | 49.8% | 50.0% | 100% |
| RLT | Five-state | 20.3% | 19.7% | 100% |
| GRU | Five-state | 20.0% | 20.1% | 100% |

All runs ended with training loss at chance (0.693 parity, 1.609 five-state). Neither model learned, even at the training length.

### Positive control

To rule out a broken training path: final-only parity at train length 4, 1,000 steps, one seed.

| Model | Eval at 4 | Eval at 8 (final / per-position) |
| --- | --- | --- |
| GRU | 100% | 100% / 100% |
| RLT | 100% | 74.6% / 91.9% |

Final-only training works; length 32 is too hard at this budget.

### Findings

- **Supervision is decisive at this budget.** Removing per-position labels took both models from near-perfect to chance.
- **It does not explain the proof-of-concept.** Its GRU reached 100%, which final-only training at this budget cannot do, so it must have used denser supervision, a curriculum, or much more training. The gap between our RLT and theirs remains unexplained.
- **No RLT-vs-GRU comparison is possible** when both are at chance.
- **A lead, not a result:** in the length-4 control, the GRU recovered running parity at every position from the final label alone, while RLT generalized only partly to length 8. Single seed, pilot scale.

### Caveats

- One budget (3,000 steps, batch 64). A curriculum or much longer training might succeed.
- The positive control is one seed at a much shorter length.