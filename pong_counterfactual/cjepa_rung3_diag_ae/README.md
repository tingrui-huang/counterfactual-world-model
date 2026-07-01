# Rung 3 diagnostic — reconstruction-AE vs Discrete-JEPA representation

**Read-only w.r.t. all frozen artifacts.** This subdirectory answers ONE narrow
question raised after the Discrete-JEPA tokenizer failed its Stage-2 gate:

> Does a simple reconstruction-trained conv encoder preserve the Pong player-paddle
> position and realized paddle displacement **better** than the frozen Discrete-JEPA
> representation?

It is a **diagnostic**, not an architecture choice, and it deliberately stops short of
training any counterfactual transition model.

## What it does

Trains a small conv autoencoder on Pong frames (pixels + reconstruction only), freezes
the encoder, then runs frozen-feature probes comparing three representations on the
**same** fixed eval pool and **same** episode-grouped split the existing Rung 3 audit
used (`cjepa_rung3/diagnose_representation.py`):

| Representation | Feature |
|---|---|
| Object state (oracle) | ground-truth 4-vector — reference/ceiling |
| Discrete-JEPA | `tokenizer_M4_K64_res84` one-hot tokens (downstream form) + pre-quant latent (best case) |
| Reconstruction AE | frozen conv-AE 16×6×6 spatial latent trained here |

## Gates

| Gate | Question |
|---|---|
| D0 | data valid (labels present, episode split, no leakage, branches grouped) |
| A0 | AE can overfit a tiny subset (recon tracks paddle, not constant-frame collapse) |
| A1 | small training reconstructs held-out frames incl. the paddle |
| P1 | frozen latent preserves single-frame `player_y` |
| P2 | two frozen frames preserve realized displacement `Δy = player_y_t1 − player_y_t` |
| P3 | factual vs counterfactual branch outcomes separable, paddle-localized |

## Objective note (evidence-based deviation)

The spec preferred **L1** reconstruction. A controlled tiny-overfit test showed L1
**mean-collapses** the sparse bright paddle on these frames (needs >4000 steps to
overfit 12 frames; recon/target variance ratio 0.78), whereas **MSE** captures it in
<1500 steps (ratio 0.997). To give the reconstruction baseline a *fair* shot at
preserving paddle position, the AE defaults to **MSE** (`loss="mse"`); `loss="l1"`
remains selectable and the collapse is documented, not hidden.

## Run

```bash
# full diagnostic (CPU, ~15-20 min at 3000 steps)
python -m pong_counterfactual.cjepa_rung3_diag_ae.run_diagnostic \
    --out pong_counterfactual/cjepa_rung3_diag_ae/outputs/main --steps 3000

# tests
python -m pytest pong_counterfactual/cjepa_rung3_diag_ae/test_diag_ae.py -q
```

Outputs (checkpoints, galleries, metrics) land under `outputs/` (git-ignored;
regenerate via the command above). The frozen tokenizer + dataset under
`pong_counterfactual/results/` are only ever read.

## Files

- `common.py` — shared data contract (reuses `cjepa_rung3.data` loaders + the audit's
  split/CI/physical-distance helpers for exact alignment).
- `ae_model.py` — the small conv autoencoder (spatial latent).
- `train_ae.py` — training + gates A0/A1 + galleries.
- `probes.py` — frozen-feature regression/classification probes with shuffled controls.
- `run_diagnostic.py` — orchestrates D0→P3, writes all artifacts + comparison/gate tables.
- `test_diag_ae.py` — focused unit tests.
