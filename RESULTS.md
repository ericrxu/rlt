# Results

All experiments use per-position state labeling, not the paper's next-token objective (5.1).

| Date | Commit | Config | Seed | What was tested | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-30 | dba0055 | parity_overfit | 0 | Overfit gate: 100 fixed programs, length 32 | Train 100%, eval@32 100% | Plateau at ln 2 until ~step 100, then sudden drop. Grad spike 9.7 at step 111, clipped. Gate stable ~0.545. |
| 2026-09-30 | ef3b093 | parity_overfit_gru | 0 | Overfit gate | Train 100%, eval@32 100% | Learned the rule. Plateau at ln 2 until ~step 105, then sudden drop. |
| 2026-09-30 | ef3b093 | parity_overfit_transformer | 0 | Overfit gate | Train 100%, eval@32 49% | Memorized, did not generalize. Per-position eval 60%. Slow, noisy convergence (~1,300 steps), grad spikes to 9.5. |