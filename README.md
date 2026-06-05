# counterfactual-world-model

Counterfactual world-model experiments on OCAtari environments.

The current line builds a learned transition model that abducts exogenous noise
via a Gumbel-Max SCM and reproduces a seed-replay oracle counterfactual more
accurately than a no-abduction intervention baseline (single-step, ground-truth
object states). See [`pong_counterfactual/cjepa_rung1/README.md`](pong_counterfactual/cjepa_rung1/README.md)
for details on the first rung.

## Layout

- `pong_counterfactual/` — counterfactual world-model code
  - `cjepa_rung1/` — Rung 1: Gumbel-Max abduction vs. intervention baseline against a seed-replay oracle

## Setup

The OCAtari environment is a dependency and is **not** vendored in this repo.
Install it as an editable clone alongside the Python deps:

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. OCAtari (editable clone)
git clone https://github.com/k4ntz/OC_Atari.git
pip install -e ./OC_Atari
# (or simply: pip install ocatari)
```

## Quick start

```bash
python -m pong_counterfactual.cjepa_rung1.collect      # collect transitions
python -m pong_counterfactual.cjepa_rung1.eval_rung1   # evaluate counterfactual accuracy
```
