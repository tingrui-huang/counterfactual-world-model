# CausalJEPA — Rung 2: abduction under the environment's OWN noise (ALE sticky actions)

Rung 2 re-runs Rung 1's abduction-vs-intervention test, but the stochasticity is no
longer self-injected wind. It is **ALE's built-in sticky actions**
(`repeat_action_probability = s`, Machado et al.) — noise that is internal to the
simulator, un-injected, un-recorded, and never shown to the model. The model still sees
only `(state, history(N), intended_action, next_state)`, does **Gumbel-Max abduction**,
and must reproduce a seed-replay **oracle** counterfactual better than a no-abduction
**intervention** baseline.

This buys two things over Rung 1:

1. **Removes the "you made the noise yourself" critique.** The noise is the environment's
   standard, un-injected stochasticity, not something we designed and recorded.
2. **An identifiability probe.** The true mechanism is a sticky Bernoulli over the
   *executed action* — **not** a Gumbel-Max draw over outcome bins. The model nonetheless
   ASSUMES a Gumbel-Max SCM. *Does abduction under a mis-specified mechanism still recover
   the true counterfactual?* The seed-replay oracle answers this empirically.

## Run

```
# 1. THE GATE (run first): is ALE sticky-action seed replay valid? (Invariant 3)
python -m pong_counterfactual.cjepa_rung2.sticky_probe

# 2. the experiment: train history(N) models, sweep s, run checks A-E
python -m pong_counterfactual.cjepa_rung2.eval_rung2
```

(use the repo `.venv`: `.venv/Scripts/python.exe -m ...`)

## Two ways this build deviates from the skill (with evidence)

| Topic | Skill locked | What we did & why |
|-------|--------------|-------------------|
| **frameskip** | **1** (Markov argument) | **2.** At fs=1 the Pong paddle has a **1-frame actuation latency**: `player_y[k+1] = player_y[k] + velocity[k]`, and `velocity[k]` is set by `action[k-1]`. So **action[k] has 0% effect on state[k+1]** (measured: 0/21 states) — the *single-step* paddle counterfactual is **degenerate** and there is nothing for abduction to recover. fs=2 is the **minimum** frameskip with a 100% one-step action effect (40/40), with the fewest hidden sub-step frames → cleanest s=0 control. The residual 2-frame paddle momentum is exactly what `history(N)` absorbs; the s=0 collapse control (check C) validates it empirically. |
| oracle route | seed replay + gate | **seed replay; the gate PASSED** (see below), so no ALE-instrument fallback was needed. |

The fs=1 degeneracy is not hypothetical — at fs=1 the gate's two-probe stuck detector reads
**1.0 everywhere** (every pair of distinct actions yields an identical next state), which is
latency, not stickiness. fs=2 fixes it. This was caught *before* trusting any number,
exactly as the gate is meant to.

## The gate (Invariant 3): seed replay validly pins the sticky draw

The whole experiment rests on one assumption: with sticky on, the ALE trajectory is STILL a
deterministic function of `(seed, intended-action sequence)`, because the sticky RNG draw is
seeded by `reset(seed)` and is **action-independent**. `sticky_probe.py` verifies this:

| s | GATE 1 (replay reproduces factual) | sticky-fire rate (floor-corrected) | GATE 3 (action-independent) |
|------|------|------|------|
| 0.00 | 100.0% (n=606) | 0.000 | 100.0% |
| 0.10 | 100.0% (n=609) | 0.097 | 100.0% |
| 0.25 | 100.0% (n=595) | 0.315 | 100.0% |
| 0.50 | 100.0% (n=600) | 0.495 | 100.0% |

**GATE 1 = 100%** is the decisive check — it *is* the Invariant-3 oracle no-op test (replaying
`a_k` reproduces the factual next-state). The sticky-fire rate cleanly tracks `s` once the
coincidence floor (action-insensitive boundary steps, measured at s=0) is subtracted, and the
draw is action-independent. **Gate passed → seed replay pins the sticky outcome.**

## Results (paddle `player_y` L1 vs the seed-replay oracle CF; frameskip=2, N=8)

**A. Abduction + oracle consistency** — both 100% at every `s` (model no-op CF reproduces the
observed token; oracle no-op replay reproduces the factual next-state).

**B. Calibration** — the model learns the sticky *mixture*: as `s` grows, the factual outcome
becomes a genuine coin-flip between "your move" and "the previous action", so held-out
`player_y` top-1 falls roughly in step with the sticky mass:

| s | player_y top-1 | (expected if mixture) |
|------|------|------|
| 0.00 | 92.8% | ~deterministic |
| 0.10 | 75.5% | |
| 0.25 | 67.0% | |
| 0.50 | 45.8% | ~half the time the action sticks |

**C + D. Headline** — `error_CF = |model_CF − oracle_CF|`, `error_IV = |model_IV − oracle_CF|`
on `player_y` (4-number L1 secondary). s=0 is the **collapse control** (no noise → CF≈IV); the
headline is the noise regime s>0:

| s | error_CF | error_IV | gap (IV−CF) | CF<IV? |
|------|----------|----------|-------------|--------|
| 0.00 | 0.558 | 0.510 | −0.048 | ctrl (C: collapse ✓) |
| 0.10 | 0.928 | 1.080 | **0.152** | yes |
| 0.25 | 1.328 | 1.524 | **0.196** | yes |
| 0.50 | 1.980 | 2.412 | **0.432** | yes |

→ **error_CF < error_IV at every s>0, and the gap grows with s** (0.152 → 0.196 → 0.432).
At s=0 the gap is −0.048 (noise around zero) — the textbook collapse control, exactly as in
Rung 1's synthetic check.

**E. The advantage concentrates on sticky-fired steps** — paddle error split by the ORACLE
two-probe `stuck` label (never a model input). On steps where the action actually stuck, the
no-abduction baseline wrongly re-rolls and "un-sticks" the move; abduction holds it:

| s | #stuck | stuck: CF / IV / **gap** | free: CF / IV / gap |
|------|------|------|------|
| 0.00 | 6 | 0.08 / 0.08 / **0.00** | 0.57 / 0.52 / −0.05 |
| 0.10 | 26 | 1.19 / 1.81 / **0.62** | 0.90 / 1.00 / 0.10 |
| 0.25 | 59 | 1.53 / 2.23 / **0.69** | 1.26 / 1.31 / 0.04 |
| 0.50 | 128 | 1.87 / 2.51 / **0.64** | 2.09 / 2.31 / 0.21 |

→ The abduction advantage is **~3–15× larger on sticky-fired steps** than on free steps. This
is the Rung-2 analogue of Rung 1's wind-fired concentration — and here, unlike Rung 1 on
real Pong, it is **clean and unambiguous**, because at fs=2 the action genuinely moves the
paddle in one step, so the sticky/free dissociation is not blurred by the paddle latency.

## The readout

On the environment's OWN sticky-action noise — never injected, never recorded, never shown to
the model — abduction reproduces the seed-replay oracle counterfactual with error **1.41** vs
the no-abduction baseline's **1.67 > it** (mean over s>0), the gap **growing with the sticky
probability** (0.15 → 0.20 → 0.43) and **concentrating on the steps that actually stuck**
(stuck-gap ≈ 0.65 vs free-gap ≈ 0.1). So Gumbel-Max abduction recovers the true counterfactual
under **real, mis-specified discrete noise** — the identifiability question, answered
empirically: even though the true mechanism is a sticky Bernoulli over executed actions and
the model assumes a Gumbel-Max SCM over outcome bins, abducting the observed transition still
pins the counterfactual better than re-rolling the noise.

## Invariants (all upheld)

1. **Model never sees the noise, the seed, or the executed action** — only
   `(state, history(N), intended_action, next_state)`. The sticky outcome is *inferred*. ✓
2. Abduction uses the actually-observed transition (teacher forcing). ✓
3. **No-op consistency validates the oracle** — model no-op CF = observed token (100%) AND
   oracle no-op replay = factual next-state (100%). The gate (`sticky_probe.py`) proves seed
   replay pins the sticky draw before any number is trusted. ✓
4. **s=0 collapse** — with sticky off and history(N) fed, `error_CF ≈ error_IV` (|Δ|=0.048),
   both small. ✓
5. **Oracle holds the same sticky outcome fixed** — replay to k with the same seed + intended
   prefix automatically applies the same sticky draw at k (action-independent), so it executes
   `a_{k-1}` if it stuck or `a'` if it did not — the same outcome as the factual. ✓

## Files

- `sticky_probe.py` — **the gate.** Verifies ALE sticky-action seed replay is deterministic,
  action-independent, and fires at ~rate s. Run this first.
- `collect_rung2.py` — env config (fs=2, `repeat_action_probability=s`, wind OFF), behavior
  policy, by-episode logging with seed + intended actions (no executed/wind — sticky is
  internal). Reuses `history.py`.
- `oracle_rung2.py` — seed-replay oracle: `cf_next_state`, `factual_next_state`, and the
  two-probe `stuck_at` label for check E.
- `eval_rung2.py` — the s-sweep + checks A–E.

Reuses from Rung 1: `gumbel_abduction.py`, `discretize.py`, `history.py` (`hist_features`,
`valid_ks`, `HistModel`), `model.py` (`POS_NORM`/`VEL_NORM`), and the `PongEnv` (now with an
optional `repeat_action_probability`).

## Out of scope (later rungs)

No learned encoder/pixels (Rung 3), no RL / sample-efficiency (Rung 4), no multi-step rollout
(Rung 1.5), no self-injected wind (that is Rung 1), no LLM.
