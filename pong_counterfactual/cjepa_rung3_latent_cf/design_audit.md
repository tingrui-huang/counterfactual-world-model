# Design audit — transferring Rung 2 counterfactuals to the paddle-local AE latent

Read-only audit of the two things being joined. Nothing here modifies a frozen artifact.

## A. Reusable Rung 2 skeleton (`cjepa_rung2/eval_rung2.py` + `cjepa_rung1/*`)

- **State representation:** ground-truth object 4-vector `(ball_x, ball_y, player_y,
  enemy_y)` via `noise.state_vec`. `player_y` (dim 2) is the headline.
- **Transition model:** `history.HistModel` — **4 independent per-dim MLP categoricals**
  (`sklearn MLPClassifier((64,))`) over discretized **delta-bins** (`discretize.Discretizer`
  fits per-dim delta vocabularies). Input = `hist_features` over an N=8 window: current
  4 positions, one-step velocities, last-N player_y, last-N intended actions (one-hot).
- **Action encoding:** intended-action one-hot over {NOOP,UP,DOWN}, per history step; the
  **CF overrides ONLY the step-k action** (`override_action=a'`), sharing all past history.
- **Discrete noise variable:** the **Gumbel-Max SCM exogenous noise** `g`
  (`gumbel_abduction.py`, Oberst & Sontag 2019). Each per-dim categorical is
  `token = argmax_i(logits_i + g_i)`, `g ~ Gumbel(0,1)`. The noise is a per-dim, per-category
  Gumbel vector — it is NOT the raw ALE sticky bit; the categorical *learns* the sticky
  behavior (mass on the delta=0 "stay" bin), and abducting `g` recovers whether this
  instance took the sticky branch.
- **Abduction:** `gumbel_posterior(logits_factual, observed_token)` — truncated-Gumbel
  posterior of `g` given the factually observed token won the argmax.
- **IV baseline:** `intervention_token(logits_cf, rng)` — swap action → new logits, draw
  **fresh** Gumbel noise (no abduction).
- **CF:** `cf_token(logits_cf, g)` — swap action → new logits, reuse the **abducted** `g`.
- **Oracle (`oracle_rung2.Oracle`):** deterministic **seed replay**:
  `reset(seed); step intended[0..k-1]; step a'`. Same seed + same `s` + same prefix ⇒ the
  sticky draw at k is identical ⇒ the SAME exogenous realization is held fixed across the
  intervention. `stuck_at` = two-probe oracle label (ORACLE-ONLY, never a model input).
- **Metric:** `error_CF/IV = |model_next − oracle_cf|` on `player_y` (headline), 4-vec L1
  (secondary). CF must beat IV, gap grows with `s`, advantage concentrates on stuck steps.
- **Tuple filtering / splits:** move-steps only (`a_int ∈ {UP,DOWN}`) with a valid N-window;
  250 samples via `rng(7)`; train episodes (seeds 0..39) disjoint from eval (1000..1013).

## B. Frozen AE / P2 follow-up identity

- **Checkpoint:** `cjepa_rung3_diag_ae/outputs/main/A1/ae_checkpoint.pt` (`ConvAutoencoder`,
  MSE, trained at git `b760106`). Encoder frozen.
- **Latent shape:** `encode` → `(N,16,6,6)` spatial map (no pooling).
- **Paddle-local region (fixed, from the successful P2 follow-up):** the **right 2 of the 6
  latent columns**, `z[:, :, :, 4:6]` → `(16,6,2)` = **192** dims. Imported verbatim from
  `p2_probe_followup.run_followup.PADDLE_COLS = slice(4,6)`; D0 asserts identity. Region is
  FIXED (player-paddle side, per P3 locality), never chosen per-sample from future labels.
- **Preprocessing:** `frames.preprocess` (grayscale, crop rows [34,194), bilinear resize to
  84). Latents from `frames_u8 → /255*2−1 → encoder`.
- **Dataset / split:** `results/data_s0.5/` (res84, s=0.5, fs=2). Eval = the fixed 250-tuple
  pool + `oracle_cache.npz` (o_fac_frame, o_cf_frame, stuck) from the AE diagnostic. Same
  episode-grouped split.

## C. Can the Rung 2 noise be reused in latent space? — NO directly; faithful adaptation

**The Gumbel-Max per-dim categorical noise CANNOT be ported unchanged.** It is defined over
a *discrete categorical outcome* (per-dim delta-bin). The primary target here is the
**continuous 192-dim latent** `z_t1_local`, which has no per-dim categorical structure, and
the task forbids decoding the latent back into object coordinates (which is the only way to
recover a categorical). So `argmax(logits+g)` has nothing to act on.

**Faithful adaptation (documented, NOT a silent change of meaning).** We keep the exact
Gumbel-Max **SCM abstraction** — *a discrete exogenous noise selects the outcome; abduct it
from the factual, reuse it under the intervention* — and only change the *parameterization*
of the mechanism to suit a continuous target:

    discrete noise  u ∈ {0,…,K−1}   (K=2, the natural sticky cardinality: took-sticky vs not)
    mechanism       z_t1_local = f_θ(z_t_local, a_int, a_prev, u)     (deterministic given u)
    abduction       û = argmin_u  ‖ f_θ(z_t_local, a_int, a_prev, u) − z_t1_local ‖²_norm
    counterfactual  z_cf = f_θ(z_t_local, a', a_prev, û)              (reuse û, swap a_int)
    intervention    z_iv = f_θ(z_t_local, a', a_prev, ũ),  ũ ∼ p(u)   (fresh noise, no abduction)

This is the discrete-choice SCM analog of Gumbel-Max: `argmin_u` over K mechanisms is the
continuous-target counterpart of `argmax_i(logits+g)` picking the winning category. The noise
still means **the exogenous sticky realization** (did the action stick), of the same binary
cardinality Rung 2's abduction effectively recovered for `player_y`. `a_prev` is an input
(as in Rung 2's history) so the "stuck ⇒ repeat previous action" branch can transfer across
the swapped `a_int` — under CF, `a_prev` is unchanged, exactly as the seed-replay oracle.

**Training the discrete noise WITHOUT oracle labels** (Rung 2 never fed the stuck label):
`u` is a latent mixture index learned by **hard-EM** (E-step = the same argmin abduction;
M-step = fit `f_θ` on the assigned `u`). Initialized from label-free latent-displacement
magnitude (‖z_t1−z_t‖: small≈stuck / large≈moved), then refined. The oracle stuck label is
used ONLY to *evaluate* abduced-`u` accuracy, never for training.

**Pre-registered choices (fixed before seeing CF results):** distance = normalized latent
MSE over the 192 local dims (same as the training loss and the abduction metric); IV = fresh
`u ∼ p(u)` where `p(u)` = empirical abduction frequency on train (matches Rung 2's "fresh
noise"); K = 2; region = `slice(4,6)`. The physical readout uses a frozen Ridge position
probe (local latent → player_y), fit on train latents, used for EVALUATION only — never a
transition-model training loss.

**Expected-limitation note (honest, pre-results):** the P2 follow-up showed the local latent
decodes movement *direction* well but *magnitude* poorly (position readout ~5px > 4.6px
paddle step). So a genuine CF>IV advantage is most likely to appear in **direction / stuck
steps**, and the latent-MSE / position magnitude advantage may be small or mixed — which the
task's "PASS WITH WARNINGS" bucket anticipates.
