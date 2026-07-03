"""Orchestrate the oracle-noise diagnostic: O1, then O2 if O1 passes, then REPORT +
gate_table. Assumes train_oracle.py produced outputs/main/oracle_ckpt.pt.
Run: python -m ...oracle_noise_diag.run_oracle
"""
from __future__ import annotations

import json
from pathlib import Path

from pong_counterfactual.cjepa_rung3_latent_cf.oracle_noise_diag import evaluate_oracle as EO

OUT = Path(__file__).resolve().parent / "outputs" / "main"


def _load(n):
    p = OUT / n
    return json.loads(p.read_text()) if p.exists() else None


def main():
    if not (OUT / "oracle_ckpt.pt").exists():
        print("no oracle_ckpt.pt — run train_oracle first."); return
    d0 = _load("d0.json"); tiny = _load("tiny_known_answer.json")
    o1 = EO.evaluate_o1()
    o2 = EO.evaluate_o2() if o1["pass"] else {"gate": "O2", "verdict": "NOT RUN",
                                              "pass": False}
    _gate_table(d0, tiny, o1, o2)
    _report(d0, tiny, o1, o2)
    print(f"\n[run] D0={d0['pass']} O1={o1['pass']} O2={o2.get('verdict')}")
    print(f"[run] report -> {OUT/'REPORT.md'}")


def _gate_table(d0, tiny, o1, o2):
    rows = [
        ("D0 oracle-noise validity", "PASS" if d0["pass"] else "FAIL",
         f"trusted two-probe stuck (train rate {d0['train_stuck_rate']:.3f}); eval "
         f"{d0['eval_stuck_counts']}; cf==fac on stuck; aligned; wrong/shuffled differ"),
        ("O1 factual w/ oracle noise", "PASS" if o1["pass"] else "FAIL",
         "; ".join(o1["reasons"])),
        ("O2 oracle-noise CF upper bound", o2.get("verdict", "NOT RUN"),
         "; ".join(o2.get("reasons", []))),
    ]
    md = ["# Gate table\n", "| Gate | Result | Evidence |", "|---|---|---|"]
    for g, r, e in rows:
        md.append(f"| {g} | {r} | {e} |")
    (OUT / "gate_table.md").write_text("\n".join(md), encoding="utf-8")


def _fv(a, k):
    return a[k]["mean"] if a and k in a else float("nan")


def _route(o2):
    v = o2.get("verdict", "NOT RUN")
    if v == "ORACLE-NOISE SUCCESS":
        return "Focus next on learned noise abduction / noise parameterization."
    if v == "ORACLE-NOISE FAILURE":
        return ("Stop improving hard-EM / abduction; revisit the latent precision or the "
                "forward-transition formulation.")
    return ("Result inconclusive. Cheapest resolving experiment: rerun O1/O2 with a "
            "higher-capacity forward head OR a sharper (higher-res / 2-frame) paddle-local "
            "latent, holding this exact oracle-noise harness fixed.")


def _report(d0, tiny, o1, o2):
    a1 = o1["aggregate"]; a2 = o2.get("aggregate", {})
    final = "PASS" if o2.get("verdict") == "ORACLE-NOISE SUCCESS" else (
        "PASS WITH WARNINGS" if o2.get("verdict", "").startswith("MIXED") else "FAIL")

    def mrow(lab, k):
        if not a2 or f"mse_{k}" not in a2:
            return f"| {lab} | — | — | — | — | — |"
        g = lambda p: _fv(a2, f"{p}_{k}")
        return (f"| {lab} | {g('mse'):.3f} | {g('dir'):.3f} | {g('bal'):.3f} | "
                f"{g('pos'):.2f} | {g('betteriv'):.3f} |")
    md = f"""# Oracle-noise upper-bound diagnostic — final report

## Goal check
- **Main question:** given the TRUE environment noise (trusted two-probe `sticky_fired`),
  can the frozen paddle-local AE latent + the small forward model produce accurate
  single-step counterfactuals?
- **Why it matters:** the parent experiment's learned-noise CF failed T2. This separates
  *learned-noise-abduction* failure from *latent/forward* failure by handing the model the
  correct noise.
- **Decision after a positive result:** the representation/forward is sufficient; the
  bottleneck is learned noise inference/parameterization → focus there next.
- **Decision after a negative result:** even known noise is insufficient → stop improving
  hard-EM/abduction; revisit latent precision or the forward-transition formulation.
- **Not being tested:** learned abduction, general causal representation, large visual world
  model, multi-step rollout, planning, RL.

## Identity (validation)
- AE checkpoint git `{d0['ae_train_git_commit'][:10]}` (frozen, deterministic — adapter D0).
- Dataset `data_s0.5` res84; same eval pool (250) + episode split; oracle noise from the
  simulator two-probe (reset-replay), NOT paddle-movement inferred; train labels aligned +
  cached; wrong/shuffled controls verified genuinely different. Prior artifacts untouched.

## Gates
| Gate | Result | Evidence |
|---|---|---|
| D0 oracle-noise validity | {'PASS' if d0['pass'] else 'FAIL'} | d0.json |
| O1 factual transition with oracle noise | {'PASS' if o1['pass'] else 'FAIL'} | oracle_factual_metrics.json |
| O2 oracle-noise CF upper bound | {o2.get('verdict','NOT RUN')} | oracle_cf_metrics.json, oracle_cf_*.png |

### O1 factual (oracle noise vs controls), mean over seeds
- oracle-noise MSE **{_fv(a1,'mse_oracle'):.3f}** vs no-noise {_fv(a1,'mse_no_noise'):.3f},
  wrong-noise {_fv(a1,'mse_wrong_noise'):.3f}, wrong-action {_fv(a1,'mse_wrong_action'):.3f},
  copy-current {_fv(a1,'mse_copy_current'):.3f}, shuffled-next {_fv(a1,'mse_shuffled_next'):.3f}.
- direction acc {_fv(a1,'dir_acc'):.3f} (majority {_fv(a1,'dir_majority'):.3f}), balanced
  {_fv(a1,'bal_acc'):.3f}, position err {_fv(a1,'pos_err'):.2f}px.

## Main decision table (O2, mean over seeds)
| Method | Latent err ↓ | Direction acc ↑ | Balanced acc ↑ | Position err ↓ | Beats-IV frac |
|---|---:|---:|---:|---:|---:|
{mrow("Oracle-noise CF", "oracle_cf")}
{mrow("IV", "iv")}
{mrow("Copy factual next", "copy_fac_next")}
{mrow("Copy current", "copy_current")}
{mrow("Wrong noise", "wrong_noise")}
{mrow("Shuffled noise", "shuffled_noise")}

Subset breakdown + copy-baseline breakdown in `oracle_cf_metrics.json`; plots
`oracle_cf_overview.png` (paired / by-subset / noise-usefulness), `oracle_cf_confusion.png`
(oracle-CF / IV / copy — note the `stay` column), `oracle_cf_gallery.png`.

## Final route decision
**{_route(o2)}**

## Final verdict: **{final}**

The frozen paddle-local AE latent {'DOES' if final=='PASS' else 'does NOT cleanly'} support
accurate single-step counterfactuals even when the true environment noise is supplied. See
the decision table and subset/copy breakdowns for the exact conditions. This diagnostic
establishes nothing about learned abduction, causal representation, world models, multi-step,
planning, or RL.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
