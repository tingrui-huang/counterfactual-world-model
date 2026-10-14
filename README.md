# Counterfactual World Modeling for Reinforcement Learning

## Overview

This project investigates counterfactual dynamics modeling for reinforcement learning:
given an observed transition, can a learned model predict what would have happened under
a different action while holding the realized environment stochasticity fixed?

The longer-term motivation is sample-efficient reinforcement learning, but this project
does not yet evaluate policy learning. It asks whether counterfactual dynamics can first
be inferred, represented, and predicted reliably in a controlled stochastic environment.

OCAtari Pong provides simulator-derived object states, raw frames, and stochastic ALE
sticky actions. Experiments progress from object-state counterfactuals to learned visual
representations and latent dynamics. Both positive and negative results are reported; the
current latent-space counterfactual model does not pass the final evaluation.

**Key findings.** Counterfactual abduction improves single-step predictions over
fresh-noise interventions in object-state Pong. This advantage does not reliably transfer
to the learned visual representations tested here: the discrete tokens lack stable
action-local structure, while the latent transition model fails stronger copy-based
controls.

## Research Questions

The project is organized around three questions:

1. Can a model infer the realized stochasticity from an observed transition and reuse it
   to predict a counterfactual outcome?
2. Does a learned visual representation preserve the fine, action-relevant motion
   information required for that inference?
3. When latent-space counterfactual prediction fails, is the limiting factor noise
   inference, representation precision, or the forward dynamics model?

## Methodology

A counterfactual changes the observed action while keeping the same exogenous realization;
an intervention baseline changes the action and samples fresh noise. The comparison is
therefore between reusing and resampling the stochasticity behind an observed transition.

Ground-truth counterfactuals use seed replay with the same ALE seed, sticky-action
probability, and action prefix. Factual reproducibility and action-independent
stochasticity are validated before replay is treated as an oracle. This is a checked
property of the ALE/Pong setup, not an assumption about stochastic simulators in general.

For discrete outcomes, a Gumbel-Max structural causal model abducts noise conditioned on
the observation, recomputes the distribution under the alternative action, and reuses the
noise. Evaluation includes consistency checks, fresh-noise interventions, copy and
shuffled baselines, and oracle-labelled subset analyses.

Reliable single-step counterfactual prediction is treated as a prerequisite for downstream
RL experiments; accurate factual dynamics alone are not assumed to imply useful
counterfactual predictions.

## Experiments and Findings

### Stage 1: Counterfactual Dynamics under Stochasticity

Stage 1 establishes the evaluation framework using compact object states. Deterministic
and synthetic-noise experiments show why counterfactual prediction must hold noise fixed,
after which categorical transition models perform Gumbel-Max abduction from observations.

Two complications are addressed. Action and position history expose paddle momentum that
is hidden from a single state vector. Self-injected noise is then replaced by ALE's
built-in sticky actions, with seed-replay probes validating the counterfactual oracle.

The main Stage 1 result is positive: in object-state Pong, reusing abducted noise predicts
the oracle counterfactual more accurately than resampling fresh noise. The advantage grows
with sticky-action probability and is concentrated on transitions affected by stickiness.
These results provide empirical support for single-step counterfactual abduction in the
object-state Pong setting.

### Stage 2: Representation Sensitivity and Visual Modeling

Stage 2 asks whether that result survives representation loss. A controlled sweep
quantizes object states while keeping the oracle at full resolution. As distinct motion
outcomes merge, the counterfactual advantage collapses. At coarse resolutions, copying
the factual outcome can imitate an advantage, motivating explicit collision and copy
checks for learned representations.

The visual experiments replace object states with a JEPA-inspired discrete tokenizer.
Frozen-representation audits show that it distinguishes many branch outcomes, but small
paddle movements affect most or all token slots, adjacent-frame tokens change rapidly,
and no stable paddle-local token emerges. Realized displacement remains poorly recoverable
relative to the object-state upper bound.

The conclusion is deliberately limited to the tested configuration: the JEPA-inspired
tokenizer distinguishes many visual states but does not provide sufficiently stable or
localized features for reliable action-level motion decoding.

### Stage 3: Representation Diagnostics and Latent Dynamics

Stage 3 investigates why visual counterfactual prediction remains unreliable. A small
reconstruction autoencoder represents paddle position better than the discrete tokenizer,
and branch differences localize to the paddle side of its latent map. Follow-up probes
recover useful movement-direction information from a paddle-local two-frame input.

A latent transition model on the frozen paddle-local representation infers a binary
mixture component from the factual transition and reuses it after swapping the action. It
passes the factual gate and beats the fresh-noise baseline in latent error, but remains
worse than copying the factual next latent. Physical direction accuracy is near chance and
position error is approximately one paddle step.

An additional experiment supplies oracle-derived, agent-step noise information to the
forward model, but counterfactual predictions still fail the stronger controls. This
suggests that improving noise inference alone may not resolve the remaining limitations,
although the experiment does not provide the complete per-frame exogenous noise sequence.

## Limitations and Current Status

Results are limited to single-step Pong counterfactuals; multi-step rollout, planning,
policy improvement, and RL sample-efficiency gains have not been demonstrated. Some
diagnostics use an agent-step summary of within-frame ALE stickiness rather than the full
per-frame noise sequence.

The visual models remain small and task-specific. The negative Stage 2 and Stage 3 results
identify limitations of the implementations tested here, not of JEPA methods,
reconstruction objectives, or latent world models in general.

**Current status (July 2026): Paused.** The object-state experiments provide evidence that
noise abduction improves single-step counterfactual prediction, while the visual
experiments identify limitations in the tested representations and latent transition
models. The original goal of evaluating counterfactual augmentation for RL sample
efficiency remains future work.

## Repository Structure

- [`pong_counterfactual/env.py`](pong_counterfactual/env.py) — OCAtari Pong wrapper and
  object-state extraction.
- [`pong_counterfactual/cjepa_rung1/`](pong_counterfactual/cjepa_rung1/) and
  [`cjepa_rung2/`](pong_counterfactual/cjepa_rung2/) — Stage 1 data collection,
  Gumbel-Max abduction, history models, sticky-action oracle, and evaluation.
- [`pong_counterfactual/cjepa_rung25/`](pong_counterfactual/cjepa_rung25/) and
  [`cjepa_rung3/`](pong_counterfactual/cjepa_rung3/) — Stage 2 coarsening studies,
  visual tokenizer, representation gates, and token-space evaluation.
- [`pong_counterfactual/audits/`](pong_counterfactual/audits/),
  [`cjepa_rung3_diag_ae/`](pong_counterfactual/cjepa_rung3_diag_ae/), and
  [`cjepa_rung3_latent_cf/`](pong_counterfactual/cjepa_rung3_latent_cf/) — Stage 3
  audits, autoencoder diagnostics, latent dynamics, and oracle-derived noise controls.

The implementation directories retain their original names so that imports, scripts, and
recorded reproduction commands remain valid.

## Reproduction

The reference environment used Windows, Python 3.13.1, NumPy 2.1.3, OCAtari 2.2.1,
and ale-py 0.12.0. OCAtari is not vendored. Its package metadata requires NumPy <2,
which conflicts with the reference Python 3.13 setup, so recreate that tested setup
explicitly:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps ocatari==2.2.1
```

The following Stage 1 experiments are runnable from source after installation:

```bash
python -m pong_counterfactual.cjepa_rung1.eval_rung1
python -m pong_counterfactual.cjepa_rung2.sticky_probe
python -m pong_counterfactual.cjepa_rung2.eval_rung2
```

Stage 2 coarsening is also source-runnable. The learned-visual experiments and Stage 3
diagnostics require generated datasets and checkpoints that are not tracked as source.
They must be regenerated or restored at the documented paths; a fresh clone cannot
reproduce every reported visual result with a single command.
