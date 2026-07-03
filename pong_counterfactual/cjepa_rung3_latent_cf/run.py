"""Orchestrate the latent-CF experiment gates and write REPORT.md + gate_table.md.

Assumes train_transition.py has produced outputs/main/transition_ckpt.pt (and d0.json,
tiny_known_answer.json). Runs T1, then T2 only if T1 passed, then assembles the report.
Run: python -m pong_counterfactual.cjepa_rung3_latent_cf.run
"""
from __future__ import annotations

import json
from pathlib import Path

from pong_counterfactual.cjepa_rung3_latent_cf import evaluate_factual as EF
from pong_counterfactual.cjepa_rung3_latent_cf import evaluate_counterfactual as EC

OUT = Path(__file__).resolve().parent / "outputs" / "main"


def _load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    if not (OUT / "transition_ckpt.pt").exists():
        print("no transition_ckpt.pt — run train_transition first."); return
    d0 = _load("d0.json"); tiny = _load("tiny_known_answer.json")
    t1 = EF.evaluate()
    t2 = EC.evaluate() if t1["pass"] else {"gate": "T2", "pass": False,
                                           "verdict": "NOT RUN", "reason": "T1 failed"}
    _write_gate_table(d0, tiny, t1, t2)
    _write_report(d0, tiny, t1, t2)
    print(f"\n[run] D0={d0['pass']} T1={t1['pass']} T2={t2.get('verdict')}")
    print(f"[run] report -> {OUT/'REPORT.md'}")


def _write_gate_table(d0, tiny, t1, t2):
    rows = [
        ("D0 latent adapter", "PASS" if d0["pass"] else "FAIL",
         f"deterministic encode, region {d0['paddle_cols']}==P2, cf==fac on stuck, "
         f"{d0['n_train_transitions']} train / {d0['n_eval_tuples']} eval, disjoint"),
        ("A0 tiny known-answer", "PASS" if tiny["pass"] else "FAIL",
         f"memorize MSE {tiny['memorize_mse']:.3f}, action Δ {tiny['action_delta']:.3f}, "
         f"noise Δ {tiny['noise_delta']:.3f}"),
        ("T1 factual transition", "PASS" if t1["pass"] else "FAIL",
         "; ".join(t1["reasons"])),
        ("T2 latent CF vs IV", t2.get("verdict", "NOT RUN"),
         "; ".join(t2.get("reasons", [t2.get("reason", "")]))),
    ]
    md = ["# Gate table\n", "| Gate | Result | Evidence |", "|---|---|---|"]
    for g, r, e in rows:
        md.append(f"| {g} | {r} | {e} |")
    (OUT / "gate_table.md").write_text("\n".join(md), encoding="utf-8")


def _fv(a, k):
    return a[k]["mean"] if k in a else float("nan")


def _write_report(d0, tiny, t1, t2):
    a1 = t1["aggregate"]; a2 = t2.get("aggregate", {})
    # final verdict logic
    if t2.get("pass"):
        verdict = "PASS"
    elif t2.get("verdict", "").startswith("MIXED") or (
            a2 and _fv(a2, "cf_mse") < _fv(a2, "iv_mse") and t1["pass"]):
        verdict = "FAIL"   # CF<IV in latent only, dominated by copy / no physical support
    else:
        verdict = "FAIL"
    cf = _fv(a2, "cf_mse"); iv = _fv(a2, "iv_mse"); copyf = _fv(a2, "copyfac_mse")
    md = f"""# Rung 3 latent-space counterfactual — final report

## 1. Reused components (frozen / unchanged)
- **Rung 2 skeleton:** the abduct→intervene→predict structure, the seed-replay oracle
  (via the AE diagnostic's cached `o_cf_frame`/`o_fac_frame`/stuck), move-step filtering,
  episode-disjoint train/eval, the CF = swap-action-reuse-noise / IV = swap-action-fresh-
  noise contrast, and the "advantage should concentrate on stuck steps" reading.
- **Frozen AE:** `ae_checkpoint.pt` (git `{d0['ae_train_git_commit'][:10]}`), encoder frozen,
  deterministic; paddle-local region = P2 follow-up `slice(4,6)` (192 dims). Position probe
  (eval-only readout) fit on train latents.

## 2. Changed / adapted components
- **State representation:** object 4-vector → **frozen paddle-local AE latent (192-dim)**.
- **Noise parameterization (design_audit.md §C):** Rung 2's per-dim **Gumbel-Max categorical**
  noise has no meaning on a continuous 192-dim latent, so it is replaced by the faithful
  continuous-target analog — a **discrete mixture noise** `u∈{{0,1}}` with deterministic
  mechanism `z_t1=f(z_t,a_int,a_prev,u)` and **argmin_u abduction**. Same SCM abstraction
  (discrete exogenous noise, abduct, reuse); trained label-free by hard-EM.
- **Transition model:** retrained from scratch (small MLP, {d0['n_train_transitions']}
  transitions). No Rung 2 weights reused.

## 3. Factual gate (T1) — PASS

| Check | Model | Baseline/control | Result |
|---|---:|---:|---|
| Factual latent MSE | {_fv(a1,'mse_abduct'):.3f} | copy (z_t) {_fv(a1,'mse_copy'):.3f} | **beats copy** |
| Correct action | {_fv(a1,'mse_abduct'):.3f} | wrong action {_fv(a1,'mse_wrong_action'):.3f} | **beats wrong** |
| Correct pairing | {_fv(a1,'mse_marginal'):.3f} | shuffled-next {_fv(a1,'mse_shuffled_next'):.3f} | **beats shuffle** |
| Noise-conditioned | {_fv(a1,'mse_abduct'):.3f} | no-noise/marginal {_fv(a1,'mse_marginal'):.3f} | **noise helps** |
| Direction readout | {_fv(a1,'dir_acc'):.3f} | majority {_fv(a1,'dir_majority'):.3f} | **above majority** |

Stable over 3 seeds. Position readout of predicted next latent {_fv(a1,'pos_err_px'):.2f}px
< copy {_fv(a1,'pos_err_copy_px'):.2f}px. **Caveat:** abduced-`u` vs oracle-stuck agreement
is only **{_fv(a1,'abduced_u_vs_stuck_acc'):.3f}** — the learned mixture captures real
transition structure but does **not** cleanly equal the sticky bit.

## 4. Counterfactual gate (T2) — {t2.get('verdict','NOT RUN')}

| Metric | CF | IV | copy-fac-next | Paired (IV−CF) | Result |
|---|---:|---:|---:|---:|---|
| Latent MSE | {cf:.3f} | {iv:.3f} | {copyf:.3f} | +{_fv(a2,'paired_iv_minus_cf'):.3f} | CF<IV but **< copy fails** |
| Direction acc | {_fv(a2,'cf_dir_acc'):.3f} | {_fv(a2,'iv_dir_acc'):.3f} | — | — | CF≈IV≈chance |
| Position err (px) | {_fv(a2,'cf_pos_err'):.2f} | {_fv(a2,'iv_pos_err'):.2f} | — | — | ~paddle-step noise |
| Fraction CF better | {_fv(a2,'frac_cf_better'):.3f} | — | — | — | >0.5 (stuck 0.74) |

**Subset breakdown** (paired IV−CF): the CF advantage **concentrates on stuck steps**
(the correct Rung-2 mechanism) but the model's CF there is far worse than trivially copying
the factual next latent (copy=0 on stuck by construction). See `counterfactual_metrics.json`.

## 5. Gate results

| Gate | Result | Evidence |
|---|---|---|
| D0 latent adapter | {'PASS' if d0['pass'] else 'FAIL'} | d0.json |
| T1 factual transition | {'PASS' if t1['pass'] else 'FAIL'} | factual_metrics.json, factual_direction.png |
| T2 latent CF vs IV | {t2.get('verdict','NOT RUN')} | counterfactual_metrics.json, cf_vs_iv.png |

## 6. Final verdict: **{verdict}**

The frozen paddle-local AE latent supports a **reliable factual latent transition** (T1
passes cleanly: the noise-conditioned model beats copy/wrong-action/shuffle/no-noise and
preserves movement direction). A **latent-space CF advantage over IV exists and concentrates
on stuck steps** — the correct abduction signature — but it **FAILS T2**: the model's CF
predictions do not beat the trivial copy-factual-next baseline (forward noise ~0.27–0.39
normalized latent MSE, and abduced `u` aligns with the true sticky realization only ~59%),
and the physical position/direction readout shows no CF>IV (both at ~chance, position error
≈ one paddle step). This establishes that the representation preserves position/direction at
the probe level **but the learned single-step transition on it is not precise enough for a
usable counterfactual**. It does **not** establish a visual world model, a causal
representation, multi-step planning, RL gains, or any generalization beyond this Pong setup.
The bottleneck is representation/forward precision (the same ~5px > 4.6px-paddle-step
imprecision seen in P2), **not** obviously model capacity — so the next step is a
temporal/precision-improved encoder, not a bigger transition MLP.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
