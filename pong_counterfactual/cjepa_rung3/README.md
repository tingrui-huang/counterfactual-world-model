# CausalJEPA — Rung 3: the learned eye (Discrete-JEPA tokenizer on pixels)

Rungs 0–2.5 ran the counterfactual machinery on OCAtari's **free** ground-truth state.
Rung 3 swaps that one component for a **learned** Discrete-JEPA tokenizer (raw Pong
pixels → M discrete tokens). Everything else — the transition model shape, Gumbel-Max
abduction (`cjepa_rung1/gumbel_abduction.py`, unchanged, applied per token slot), the
seed-replay oracle, the err_CF-vs-err_IV headline, the Rung-2.5 acceptance protocol —
is structurally identical.

**The question (Problem B).** A JEPA eye abstracts by *predictability*; the exogenous
noise (did the sticky action fire? visible as a ~4.6 px paddle non-move) is by
definition the unpredictable part. Rung 2.5 quantified the stakes: the eye must
resolve finer than ~4.6 px (knee `c*=4`), and a lossy eye **fakes** the headline by
collapsing into factual-copying — so the headline alone is never sufficient.

## Stage order (0 → 1 → **2 GATE** → 3 → 4)

| stage | script | runtime | what |
|---|---|---|---|
| 0a | `stage0_collect.py` | CPU, once | Rung-2-identical rollouts (s∈{0,0.5}, fs=2, same seeds) + 84×84 frames; per-episode shards |
| 0b | `stage0_oracle.py` | CPU, once | pixel oracle: CF / factual / other-branch next **frames** per eval tuple; replay-determinism gate |
| 1 | `stage1_pretrain.py` | **GPU** | Discrete-JEPA pretrain (the only heavy step); checkpoint-resume every 500 steps |
| 2 | `stage2_gate.py` | minutes | **the gate**: collision probe + linear probe on the frozen eye; logs every config to `gate_log.json` |
| 3 | `stage3_transition.py` | light | factorized M-slot token forecaster; **refuses to run if the gate failed** |
| 4 | `stage4_eval.py` | light | abduction vs intervention vs pixel oracle; checks A–E; headline figure |

Colab: open `rung3_colab.ipynb` (mount Drive → stage cells in order). All artifacts
live under `/content/drive/MyDrive/cjepa_rung3/`; every stage loads its inputs from
that root and resumes from its last checkpoint — a disconnect costs at most one
checkpoint interval. Locally: set `CJEPA_RUNG3_ROOT` (defaults to
`cjepa_rung3/artifacts/`) and run the same modules; `--smoke` on every stage runs a
tiny CPU end-to-end pass (plumbing only, not science).

## The six decisions (as taken)

1. **Frame format** — grayscale, crop rows [34,194) (160×160 play area), resize
   **84×84** (`config.FrameSpec`). ⚠ The resize itself coarsens: paddle step 4.59 px
   → **2.41 resized px**. If the Stage-2 collision gate fails, this is the **first
   suspect** → `res=128` (or 160 = no resize) and recollect.
2. **Code base** — self-contained assembly in `tokenizer.py` (I-JEPA-style
   context/EMA-target/predictor + VQ; Discrete-JEPA has no public code). *Deviation
   from the skill's "pull `eb_jepa`" default, reasoned:* a pinned external repo inside
   a Colab notebook is a fragile moving dependency, and at this scale (≈1.5 M-param
   ViT on 84×84 frames) the recipe is small enough to own — every knob the gate may
   need (M, K, mask ratio, loss weights) sits in one file. The training recipe follows
   eb_jepa's structure: context encoder, EMA target, predictor, latent L2, **no
   reconstruction loss**.
3. **Global vs slots** — v1 = **GLOBAL** semantic tokens (frame → M tokens).
   Object-centric slots stay in reserve as the positive control if the global gate
   cannot be made to pass. (Per the skill: run v1 now, bring the collision result to
   the supervisor rather than block.)
4. **Shape** — **M=4 tokens/frame, K=64** to start; the primary gate knobs, in order:
   K↑ → M↑ → loss weights → decision-① resolution.
5. **Transition** — **factorized**: M independent K-way categoricals (one softmax per
   slot), so `gumbel_abduction.py` applies per slot with zero change.
6. **Scoring** — **dual**: token space primary (oracle CF frame → frozen eye → target
   tokens; per-slot mismatch), physical px secondary (Stage-2 linear probe decodes
   predicted tokens → player_y; comparable to Rungs 1–2.5).

## The gate (Stage 2) — the soul of the rung

- **Collision probe**: on FREE eval tuples, encode the factual next frame and the
  would-have-stuck ("other-branch") next frame through the frozen eye; collision =
  all M tokens identical. (Rung 2.5 convention — at a stuck step the not-stuck branch
  is unreachable by seed replay; free steps give the same contrast mirrored.)
- **Linear probe**: ridge, token one-hots → positions in resized px (fit train,
  scored eval). `player_y` MAE must be < 2.41 resized px (the paddle step).
- **Rule**: collision ≤ 0.25 (the Rung-2.5 knee `c*=4px` had 0.246) AND probe ok.
  FAIL → back to Stage 1, turn knobs; Stage 3 refuses to train (`--force` exists but
  burns compute on a blind eye). Every config tried = one row in `gate_log.json` —
  **that table is the Problem-B finding**, pass or fail.

## Acceptance (Rung 2.5 protocol — the headline alone is gameable)

- **A** s=0 collapse (err_CF ≈ err_IV) + no-op identity (abduction reproduces the
  observed token, per slot) + oracle replay determinism (re-confirmed in stage 0b:
  100% vec AND frame).
- **B** headline: err_CF < err_IV at s>0 (token and px).
- **C** collision rate reported alongside; the gap must live on low-collision tuples.
- **D** copy baseline ("next token = observed token") must NOT beat the CF — copy
  winning = the c=16 failure from Rung 2.5 (collapsed eye faking the headline).
- **E** stuck-split: the advantage concentrates on oracle-labelled stuck steps.

## Hard invariants (auto-fail if violated)

- The ground-truth 4-vectors in the shards are **oracle/probe-only** — they never
  reach the tokenizer or the transition model.
- The oracle replays at full pixel fidelity; only the *model* sees learned tokens.
- Everything persistent is on Drive and resume-safe (keep-last-3 checkpoints,
  temp-file-then-rename writes).
- One token config per counterfactual query; the eval tuple set is fixed once
  (`rng(7)`, Rung-2's selection) and shared across all configs.

## Files

```
config.py            decisions ①–⑥, path resolution (Drive/local), smoke overrides
frames.py            ① grayscale/crop/resize + exact px conversions
data.py              episode shards, valid_ks, the fixed eval pool
ckpt.py              resume-safe checkpointing (keep-k, torn-write fallback)
stage0_collect.py    0a — collection with pixels
stage0_oracle.py     0b — pixel oracle cache
tokenizer.py         ② the Discrete-JEPA model + load_eye/encode_frames
stage1_pretrain.py   1  — pretrain (GPU, resume-safe)
stage2_gate.py       2  — collision + linear probe, gate rule, gate_log.json
stage3_transition.py 3  — ⑤ factorized token forecaster (gate-checked)
stage4_eval.py       4  — ⑥ dual-scored abduction eval, checks A–E, figure
rung3_colab.ipynb    the Colab driver
```

Smoke-tested end-to-end locally (CPU): the deliberately under-trained smoke eye
collapses its codebook → the gate FAILS with collision 100%, and a `--force`d Stage 4
shows exactly the degenerate signature the protocol exists to catch (all-identical
tokens, headline B fails, copy ties CF). The machinery detects what it must detect.
