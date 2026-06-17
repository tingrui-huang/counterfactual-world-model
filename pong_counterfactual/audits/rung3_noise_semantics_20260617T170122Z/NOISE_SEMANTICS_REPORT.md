# Rung 3 Stage A1.2 — Noise Semantics & Realized-Motion Audit

**Verdict: DISCRIMINATIVE_ONLY**

Generated 20260617T170139Z • Checkpoint `final.pt` • read-only (no training, no Stage 3/4)

## 1. ALE sticky / frame-skip mechanics (resolves the 3.2 px puzzle)
- **Wrapper path:** `PongEnv` -> OCAtari (`gym.make`, kwargs frameskip=2,
  repeat_action_probability=0.5) -> ale_py `AtariEnv.step`, which runs
  `for _ in range(frameskip): self.ale.act(action_idx)` (ale_py/env.py:304). Sticky is
  applied **inside each `ale.act()`**, i.e. **per internal emulator frame**, not once per
  agent step.
- **What is repeated:** ALE's sticky replaces the current action with `lastAction` = the
  action ALE **last executed** (updated every frame), NOT the agent's logged intended
  action.
- **Why logged `a_prev` is always NOOP yet stuck transitions still move ~3.2 px:**
  the behavior policy issues moves as isolated pulses, so the logged intended action at k-1
  is NOOP; but the action ALE *executed* on the last frame of step k-1 can be a real move
  (sticky chaining), and that is what gets repeated. With 2 frames/step and
  per-frame stickiness, a "stuck" step typically suppresses only PART of the motion, so the
  paddle moves a partial amount (3.2 px) rather than the full free
  step (4.6 px). Empirical stuck rate
  0.51 ~= s (not s^2); the exact per-frame draw count is
  internal to ALE and was **not logged**.

## 2. The actual noise variable (noise_target_definition.json)
The exogenous noise is the per-frame sticky draw sequence; its only well-posed **observable
effect** in the logged data is the **realized paddle displacement Δy**. The binary `stuck`
bit is a lossy 2-probe summary (A1.1 showed it is weakly identifiable even from ground
truth). Per-frame repeated-action pattern/count and the realized action sequence were **not
logged and cannot be reconstructed** from agent-step states.

## 3-4. Re-posed targets x frozen features (realized_motion_results.csv)
Targets: displacement Δy, direction (3-cls), magnitude |Δy|, residual-vs-free (free branch
estimated for stuck tuples — replay-unreachable, A1.1). Decoded from: action-only,
ground-truth object transition, pixel-difference, JEPA one-hot / codebook-emb / pre-quant.
- direction: oracle(gt) **0.695**, action-only 0.639,
  best JEPA **0.576** (base 0.402).
- displacement R²: oracle **0.783**, best JEPA **-0.416**.
JEPA adds little over actions and trails the (informative) ground-truth ceiling.

## 5. Representation structure / token locality (token_locality_metrics.csv)
- PURE-paddle corr(token Hamming, |Δpaddle|) = **0.387** (branch pairs,
  same ball; moderate ⇒ Hamming partially grades but is near-saturated).
- a paddle-ONLY change flips **3.55/4** tokens, spread
  across ALL slots (no paddle-local slot); the realized step (ball-dominated) flips
  **3.03/4**.
- fraction of small (≤1 paddle step) PURE paddle moves that flip ALL 4 tokens =
  **0.708**.
- adjacent-frame token persistence = **0.26**
  (low ⇒ tokens churn frame-to-frame, ball-driven).
See `hamming_vs_displacement.png`, `per_slot_flip_rates.png`.

**Interpretation:** the discrete tokens are **global/entangled and ball-dominated** — a
paddle-only change is smeared across all 4 slots (no slot specializes in the paddle),
~half of small paddle moves flip every token, and frame-to-frame persistence is low. The eye
SEPARATES frames (low collision, high Hamming) but provides **no stable, local per-slot
transition structure** a transition model could use to abduct the realized-motion noise.
JEPA features cannot decode realized paddle displacement (R²<0, below action-only).

## Verdict: DISCRIMINATIVE_ONLY
JEPA SEPARATES frames (discriminative, low collision) but its discrete tokens flip near-globally with no stable transition structure — so realized-motion / noise-effect decoding from the transition is weak. The frames are distinguishable; the *transition representation* is not abduction-usable as-is.

## Recommended next step
Before any architecture change: the abduction signal that IS well-defined is the realized
paddle displacement. Two options follow from this audit — (a) make the eye's tokens
position-LOCAL/graded (e.g. object-centric slots or higher paddle resolution) so token
distance tracks paddle displacement, and/or (b) log the per-frame executed-action sequence
so the noise variable is fully observable. Do NOT train Stage 3 on the current global tokens.

## Gate table
| Gate | Check | PASS criterion | Result | Evidence artifact | Notes |
| --- | --- | --- | --- | --- | --- |
| N1 | ALE sticky/frame-skip mechanics + 3.2px puzzle | mechanics explained from code + data; binary stuck shown lossy | RESOLVED | NOISE_SEMANTICS_REPORT.md (sec 1), realized_motion_results.csv | frameskip=2, sticky applied inside per-frame ale.act() loop (ale_py env.py:304); repeated action = last EXECUTED frame-action (not the logged intended NOOP). Empirical stuck rate 0.51~=s=0.5 (not s^2), and stuck moves partial (3.2px) vs free full (4.6px) -> binary stuck collapses a partial, continuous motion-suppression process. |
| N2 | re-posed targets (displacement/direction/magnitude/residual) x features | JEPA decodes realized motion clearly above action-only and near oracle | WEAK | realized_motion_results.csv | direction: oracle=0.695 action_only=0.639 JEPA=0.576 (base 0.402); displacement R2: oracle=0.783 JEPA=-0.416. JEPA adds little over actions and trails oracle. |
| N3 | representation structure: token locality / transition stability | paddle-only token change grades with Δpaddle; small moves don't flip all tokens | FAIL (global/entangled tokens) | token_locality_metrics.csv, hamming_vs_displacement.png, per_slot_flip_rates.png | PURE-paddle corr(Hamming,Δpaddle)=0.387; a paddle-only change flips 3.55/4 tokens spread across ALL slots (per-slot ~0.47, no paddle-local slot); 0.708 of small paddle moves flip ALL tokens; adjacent-frame persistence 0.26 (ball-driven churn). No stable local per-slot transition structure. |
