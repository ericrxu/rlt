git# AGENTS.md — Recurrent Looped Transformer (RLT)

## Purpose

This repository implements the Recurrent Looped Transformer (RLT) from:

> Yifan Zhang, *Recurrent Looped Transformer*, technical report, September 12, 2026.
> Paper: `docs/Recurrent_Looped_Transformer.pdf`

**The paper is the specification.** This is a faithful implementation, not an improved one.
The owner of this repo is learning the architecture by building it. Your job is to implement
exactly what the paper specifies, explain your reasoning, and surface ambiguity. It is not to
optimize, simplify, or "fix" the design.

## Precedence

1. The paper.
2. This file.
3. Instructions in the current task.

If a task instruction conflicts with the paper or with an invariant below, **stop and say so
before writing any code.** If this file and the paper disagree, the paper wins, and you must
tell me about the discrepancy.

---

## What RLT is

A **causal encoder** `E_θ` processes tokens (parallelizable over positions) and produces
representations `e_t`. From `e_t` it builds a prefix-restricted global key–value memory `M_{≤t}`.

A **recurrent decoder** `D_φ` runs once per token, **in strict token order**. For each token it
merges `e_t` with the previous decoder output `s_{t-1}`, runs `L_D` decoder blocks, and produces
`s_t`, which is used to predict the next token and is fed into the next token's merge.

The decoder state carried between tokens is the **pair**:

```
H_t = (s_t, C^D_t)
```

- `s_t ∈ R^d` — the recurrent output (final decoder hidden state).
- `C^D_t` — the retained key/value projections at **every** decoder SWA layer.

The same transition is applied to every token: prompt, response, user, tool. There is **no reset**
at the prompt–response boundary. The index `T` marks a serving boundary, not a change in the model.

Important structural fact: the decoder never sees raw token embeddings. Its input at each step is
only `u_t = Merge(e_t, s_{t-1})`.

---

## Reference equations (from the paper — do not alter)

### Sequence and state (§2.1)

- `x_1 = BOS` for every independent sequence.
- `L_E`, `L_D` are encoder and decoder depths; `d` is residual width.
- `W ≥ 1` is the SWA window size. **`W` includes the current token.** After each update the cache
  retains **at most `W − 1`** positions per layer for the next update.
- `s_⋆` is a **learned** initial state, and is a model parameter.

### Encoder and memory (§2.2)

```
e_{1:T} = E_θ(x_{1:T})                                         (2.1)

k^g_t = P^g_K(e_t, t)
v^g_t = W^g_V · RMSNorm_E(e_t)
M^g_{≤t} = { (k^g_j, v^g_j) }  for j = 1..t                     (2.2)
```

- `P^g_K` includes normalization, projection, **and** any positional transformation.
- Memory groups `g ∈ {1, …, G}`. Decoder layer `ℓ` reads group `g(ℓ)`.
  `G = 1` shares memory across all decoder layers; `G = L_D` gives layer-specific projections.
- Memory depends on encoder representations only, never on decoder states.

### One transition for every token (§2.3)

```
H_0 = (s_⋆, ∅)                                                 (2.3)

u_t = Merge(e_t, s_{t-1})                                      (2.4)
H_t = (s_t, C^D_t) = D_φ(u_t ; M_{≤t}, C^D_{t-1}, t),  t ≥ 1   (2.5)
p_Θ(x_{t+1} | x_{1:t}) = softmax( W_o · RMSNorm_o(s_t) )_{x_{t+1}}   (2.6)
```

### Merge (§2.5)

```
r_{t-1} = RMSNorm_s(s_{t-1})                                   (2.9)
g_t     = σ( W_g [e_t ; r_{t-1}] + b_g )                       (2.10)
u_t     = e_t + α · g_t ⊙ (W_s r_{t-1})                        (2.11)
```

- `W_g ∈ R^{d×2d}`, `b_g ∈ R^d`, `W_s ∈ R^{d×d}`, `α` a scalar feedback scale.
- `[e_t ; r_{t-1}]` is concatenation. `⊙` is elementwise product. `σ` is the sigmoid.
- The paper does **not** prescribe an initialization for `α`. Do not choose one silently.

### Decoder block, layer ℓ (§2.5) — sublayer order is fixed

With `z^0_t = u_t`:

```
q^{D,ℓ}_t = P^{D,ℓ}_Q(z^{ℓ-1}_t, t)                                     (2.12)
k^{D,ℓ}_t = P^{D,ℓ}_K(z^{ℓ-1}_t, t)
v^{D,ℓ}_t = W^{D,ℓ}_V · RMSNorm_{S,ℓ}(z^{ℓ-1}_t)                        (2.13)

b^ℓ_t = z^{ℓ-1}_t + AttnD_ℓ( q^{D,ℓ}_t , {(k^{D,ℓ}_j, v^{D,ℓ}_j)} for j = max(1, t−W+1) .. t )   (2.14)
a^ℓ_t = b^ℓ_t     + AttnM_ℓ( P^{M,ℓ}_Q(b^ℓ_t, t) , M^{g(ℓ)}_{≤t} )     (2.15)
z^ℓ_t = a^ℓ_t     + FFN_ℓ( RMSNorm_{D,ℓ}(a^ℓ_t) )
s_t   = z^{L_D}_t                                                        (2.16)
```

Order per layer: **(1) causal SWA → (2) cross-attention to encoder memory → (3) FFN.**

- Attention operators include their output projections.
- Query/key maps include their own normalizations and positional transformations.
- **The current token's K/V at layer ℓ are formed from the layer input `z^{ℓ-1}_t` before SWA runs.**
  This is what prevents a circular dependency.
- Historical decoder K/V come from `C^D_{t-1}`.
- Alternative sublayer orders are different models. Do not reorder.

### Cache retention rule (§2.5)

After the update at position `t`, retain positions:

```
max(1, t − W + 2), …, t
```

in `C^D_t`. For `W = 1` this set is **empty**: the decoder attends only to the current token and all
cross-token decoder information flows through `s_t`.

Note the offsets: SWA **reads** `max(1, t−W+1)..t` (up to `W` entries including current), and
the cache **retains** `max(1, t−W+2)..t` (up to `W−1` entries) for the next step. These are
different on purpose. Do not "simplify" them to the same expression.

### Tied configuration (§2.6)

When `tied = True`:

- `L_E = L_D = L`.
- Encoder self-attention at layer ℓ and decoder SWA at layer ℓ **share** Q, K, V, and output
  projection matrices. Their FFNs are also shared.
- Decoder cross-attention has **separate** query/output projections, plus the memory projections of (2.2).
- Stage-specific normalizations, the merge, and the readout remain separate modules.
- This is parameter reuse, not activation copying. No decoder output is identified with an encoder output.

When `tied = False`, `E_θ` and `D_φ` have fully independent parameters. The recurrence is unchanged.

### Objectives (§5.1, §5.2)

Pretraining — every non-BOS target supervised:

```
L_PT = − E[ (1 / (S−1)) · Σ_{t=1}^{S−1} log p_Θ(x_{t+1} | x_{1:t}) ]     (5.1)
```

SFT — mask `m_{t+1} ∈ {0,1}` selects assistant targets:

```
L_SFT = − E[ (1 / Σ_t m_{t+1}) · Σ_{t : m_{t+1}=1} log p_Θ(x_{t+1} | x_{1:t}) ]   (5.2)
```

- Normalize **per example** by its number of selected targets, then average across examples.
- Examples with zero selected targets are excluded from the average.
- A token-weighted batch normalization is a different objective and must be declared explicitly.

### RL replay (§5.3, Appendix A.4)

```
r_i(Θ) = exp( log p_Θ(y_i | c, y_{<i}) − log μ(y_i | c, y_{<i}) )       (5.5)
```

- The sampler records tokens, action mask, and **actual** behavior log-probabilities `log μ`,
  including temperature, top-k/top-p truncation, and renormalization.
- The trainer rebuilds encoder features, `s_t`, **and every decoder SWA cache** from `H_0` under the
  **current** parameters, over the full prompt and response.
- Each action's log-probability is read from the **preceding** state, before that action is consumed.
- External (user/tool) tokens update the state and remain differentiable, but receive **no**
  importance-ratio factors.
- After every optimizer step, all parameter-dependent caches are invalid for exact replay.
- Old sampler hidden states are never substituted for current-policy states.

---

## Hard invariants

These are rules. Violating any of them produces a different model, even if tests appear to pass
and loss improves.

1. **Cross-attention reads only `M_{≤t}`.** At decoder position `t`, never read encoder memory for
   positions `> t`, even when the full prompt memory is already computed and in memory.
2. **SWA reads only its window.** Never read decoder K/V older than `max(1, t−W+1)` or newer than `t`.
3. **The decoder loop over time is sequential. Do not vectorize it.** A parallel SWA decoder pass is
   not equivalent to the recurrence (§2.4). Historical decoder K/V must be produced by the preceding
   recurrent updates. Batching is allowed only across **independent sequences**, never across time.
4. **The state is the pair `(s_t, C^D_t)`.** Every detach, reset, snapshot, restore, and copy must
   handle both components. Handling only `s_t` is a bug.
5. **No dropout.** Not in the model, not in tests. All tests run in `float64` with fixed seeds and
   `torch.use_deterministic_algorithms(True)`.
6. **Loss masks mask loss only.** They never disable state updates, never detach tensors, and never
   stop gradients. A loss mask is not an attention mask and is not a state reset.
7. **Independent sequences are isolated.** Separate `s_t`, separate decoder SWA caches, separate encoder
   caches, separate positions, disjoint attention masks. No attention crosses a document boundary.
8. **No reset at the prompt–response boundary.** Neither `s_t` nor `C^D_t` is reset when generation
   begins, when a user turn begins, or when a tool result arrives. Reset only at a new independent sequence.
9. **Every consumed token receives exactly one recurrent update and one K/V insertion at every decoder
   SWA layer.** No double updates, no skipped updates.
10. **Position conventions are identical** across parallel encoding, incremental encoding, prefill,
    generation, training, and replay. Position is an explicit absolute index, never inferred from a
    tensor's position in a batch.
11. **Full BPTT is the reference training computation.** Gradients flow through `s_t`, through decoder
    SWA K/V, and through encoder memory.
12. **Checkpointing is not truncation.** Activation checkpointing must produce gradients identical to
    no checkpointing. Mutable caches must be restored correctly during recomputation.
13. **SWA eviction is not a stop-gradient.** Evicted entries still influenced later states before
    eviction, and full BPTT differentiates through that.
14. **Truncation must be explicit.** Detaching only `s_t` leaves paths through decoder K/V and encoder
    memory. Any truncated-BPTT option must name every detached tensor. Running prompt recurrence
    under `no_grad` is a gradient approximation and is never the default.

---

## Rules of engagement

1. **Never modify anything in `tests/` unless I explicitly ask.** If a test fails, the code is wrong,
   not the test. If you believe a test is genuinely wrong, stop, explain why with reference to the
   paper, and wait.
2. **Never delete, weaken, loosen, skip, or `xfail` an assertion to make a suite pass.** This includes
   raising tolerances, replacing exact equality with approximate equality, and reducing test sizes.
3. **Never change a config default silently.** Propose the change with a reason and wait for approval.
4. **Explain your reasoning before writing code** for anything touching:
   cache indexing, window boundaries, masking, positions, the merge, gradient paths, detaching,
   checkpointing, or state reset. State the indices you will use and why they match the paper.
5. **Do not commit results files.** Do not write to `RESULTS.md`. I commit numbers.
6. **Do not add features, optimizations, or "improvements" that are not in the paper.** No fused
   kernels, no parallel scans, no alternative sublayer orders, no extra normalization, no dropout,
   no weight tying beyond §2.6, unless I ask.
7. **When the paper is ambiguous, stop and ask.** Do not pick a plausible interpretation and proceed.
   See "Known ambiguities" below.
8. **If a result improves unexpectedly, suspect a leak first.** Before reporting an improvement, check
   invariants 1, 2, and 7, and run the full verification suite.
9. **Do not run `git push`, `git merge`, `git rebase`, or anything that rewrites history.** You may
   stage and commit on the current branch only if I ask. I handle branches, pushes, and PRs.
10. **Keep changes small and scoped to the task.** Do not refactor unrelated files.

---

## Project choices (not specified by the paper)

These are implementation decisions for this repo. They may be changed **only with my approval**,
and any change must be recorded here.

| Choice | Current decision |
| --- | --- |
| Encoder block internals | Mirrors the decoder block without cross-attention. Per layer ℓ: `n = RMSNorm_{S,ℓ}(z)`; `q = RoPE(W_Q n, pos)`; `k = RoPE(W_K n, pos)`; `v = W_V n`; `b = z + Attn(q, k, v, causal mask)` (Attn includes `W_O`); `z' = b + FFN(RMSNorm_{F,ℓ}(b))`. No final norm on `e_t`. |
| Positional scheme | RoPE, applied from explicit absolute position index |
| FFN activation | GELU, no bias, hidden width `4d` |
| `α` learned or fixed; initial value | TBD — to be swept |
| Memory groups `G` | `1` to start |
| `tied` | `False` to start |
| Readout `W_o` tied to embeddings | TBD — paper does not specify |
| Number of heads, head dim, KV width | TBD |
| Position numbering | Absolute, starting at 1 for BOS |
| Batching | All sequences in a batch are aligned (same length, shared position counter). Variable-length batching is out of scope until Stage 12. |
| Memory normalization | One `RMSNorm_E` shared across groups, feeding both keys and values: `n_t = RMSNorm_E(e_t)`; `k^g_t = RoPE(W^g_K n_t, t)`; `v^g_t = W^g_V n_t` |
| Memory positions | RoPE on memory keys only, never on values. Cross-attention queries (Stage 6) get RoPE at the decoder's current position `t`. |
| Memory groups | Only `G = 1` (all decoder layers read group 0) or `G = L_D` (layer ℓ reads group ℓ) are supported |

## Known ambiguities — ask, don't guess

- The internal structure of encoder blocks (the paper only requires causality).
- How positional transformations apply to SWA keys vs. cross-attention queries vs. memory keys.
- Initialization of `α`, `s_⋆`, `W_s`, `W_g`, `b_g`.
- Whether `W_o` shares weights with the token embedding.
- Head counts and KV widths for SWA vs. cross-attention.
- Anything not covered above.

---

## Required tests (the contract)

These tests define correctness. They must exist and pass. You may not modify them.

- **Encoder schedule equivalence** — parallel, incremental, and chunked encoding give identical `e_t`.
- **Encoder causality** — perturbing token `j` leaves `e_i` bit-identical for all `i < j`.
- **Memory prefix immutability** — memory for a prefix equals the first entries of memory for a longer sequence.
- **Merge** — `α = 0` gives `u_t == e_t` exactly; gate lies in `(0, 1)`; gradient reaches `s_{t-1}` when `α ≠ 0`.
- **Window and cache** — `W = 1` attends only to self; cache never exceeds `W − 1` entries; correct eviction;
  caches are per-layer.
- **Serving-split invariance (Prop. 3.1)** — full prefill, split prefill + incremental at several split
  points, and fully incremental give identical `s_t`, `C^D_t`, encoder caches, and logits.
- **Whole-model causality (Prop. B.1)** — perturbing token `j` leaves `H_i` bit-identical for `i < j`,
  and changes `H_j`.
- **Gradient completeness** — `gradcheck` passes in float64 on a tiny config; detaching only `s_t` still
  lets gradient reach earlier parameters through K/V; detaching `s_t`, `C^D_t`, and encoder boundary
  tensors truncates the horizon.
- **Checkpointing parity** — gradients with and without activation checkpointing match.
- **Loss-mask independence** — supervising only the last token still gives nonzero gradient to
  parameters used at early positions.
- **Document isolation** — perturbing one sequence leaves another bit-identical.

## Commands

```
pytest tests/ -x -q          # run all tests, stop at first failure
pytest tests/test_X.py -v    # run one file verbosely
```

(Training and evaluation commands will be added as they exist.)

## Reference results

Independent proof-of-concept reported on the project page (~79K parameters, 3 seeds, trained at 32
operations, evaluated at 128 operations, 2,048 programs per task per length). Final-state accuracy at 128:

| Task | RLT | GRU | Chance |
| --- | --- | --- | --- |
| Parity | 60.8% | 100% | 50% |
| Five-state transitions | 20.7% | 99.97% | 20% |

If this implementation substantially beats these RLT numbers, treat it as a probable leak until the
full verification suite and invariants 1, 2, and 7 have been re-checked.
