"""Latent-space counterfactual vs intervention (Gate T2). Runs only if T1 passed.

CF  = abduct u from the factual transition, swap action, REUSE u.
IV  = swap action, FRESH u ~ p(u) (no abduction) — averaged over iv_noise_seeds draws.
Oracle = frozen AE latent of the seed-replay CF frame (o_cf_frame), same exogenous draw.
Compared in normalized latent space AND via the frozen position-probe physical readout.
Run: python -m pong_counterfactual.cjepa_rung3_latent_cf.evaluate_counterfactual
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf import transition as TX
from pong_counterfactual.cjepa_rung3_latent_cf.evaluate_factual import (
    load_run, abduct_predict, readout_py, _per_mse, _jd, OUT)


@torch.no_grad()
def _cf_iv(m, prior, Zt, A_fac, A_cf, Zt1, Zcf_or, n_iv, seed):
    """Return per-tuple CF and IV latent errors + predicted CF/IV latents."""
    u_hat, _ = m.abduct(TX._to_t(Zt), TX._to_t(A_fac), TX._to_t(Zt1))
    z_cf = m(TX._to_t(Zt), TX._to_t(A_cf), u_hat).numpy()
    e_cf = _per_mse(z_cf, Zcf_or)
    # IV: fresh u ~ prior, averaged over draws
    rng = np.random.default_rng(100 + seed)
    e_iv_draws, z_iv_last = [], None
    for _ in range(n_iv):
        u_iv = TX._to_t(rng.choice(len(prior), len(Zt), p=prior), long=True)
        z_iv = m(TX._to_t(Zt), TX._to_t(A_cf), u_iv).numpy()
        e_iv_draws.append(_per_mse(z_iv, Zcf_or)); z_iv_last = z_iv
    e_iv = np.mean(e_iv_draws, 0)
    return e_cf, e_iv, z_cf, z_iv_last, u_hat.numpy()


def evaluate(cfg=None):
    OUT.mkdir(parents=True, exist_ok=True)
    # T1 must have passed
    t1 = json.loads((OUT / "factual_metrics.json").read_text())
    if not t1["pass"]:
        (OUT / "counterfactual_metrics.json").write_text(json.dumps(
            {"gate": "T2", "status": "NOT RUN", "reason": "T1 failed"}, indent=2))
        print("[T2] NOT RUN — T1 failed."); return {"status": "NOT RUN"}

    models, norm, probe, cfg, ck = load_run(cfg)
    ev = LA.build_eval_tuples(cfg)
    Zt, Zt1 = norm.fwd(ev.z_t), norm.fwd(ev.z_t1)
    Zcf_or = norm.fwd(ev.z_cf_oracle)
    A_fac = LA.action_feats(ev.a_int, ev.a_prev)
    A_cf = LA.action_feats(ev.a_prime, ev.a_prev)
    dir_cf_true = np.sign(ev.py_cf - ev.py_k).astype(int)
    maj = np.bincount((dir_cf_true + 1)).argmax() - 1
    n_iv = cfg["iv_noise_seeds"]

    per_seed = []
    z_cf0 = z_iv0 = None
    for mm in models:
        m, prior, sd = mm["model"], mm["prior"], mm["seed"]
        e_cf, e_iv, z_cf, z_iv, u_hat = _cf_iv(m, prior, Zt, A_fac, A_cf, Zt1, Zcf_or,
                                               n_iv, sd)
        # copy-factual-next baseline: predict the factual next latent as the CF
        e_copyfac = _per_mse(Zt1, Zcf_or)
        # physical readout
        py_t = readout_py(probe, norm, Zt)
        py_cf = readout_py(probe, norm, z_cf); py_iv = readout_py(probe, norm, z_iv)
        dcf = np.sign(py_cf - py_t).astype(int); div = np.sign(py_iv - py_t).astype(int)
        row = {
            "seed": sd,
            "cf_mse": float(e_cf.mean()), "iv_mse": float(e_iv.mean()),
            "copyfac_mse": float(e_copyfac.mean()),
            "paired_iv_minus_cf": float((e_iv - e_cf).mean()),
            "frac_cf_better": float((e_cf < e_iv).mean()),
            "cf_dir_acc": float((dcf == dir_cf_true).mean()),
            "iv_dir_acc": float((div == dir_cf_true).mean()),
            "dir_majority": float((dir_cf_true == maj).mean()),
            "cf_pos_err": float(np.abs(py_cf - ev.py_cf).mean()),
            "iv_pos_err": float(np.abs(py_iv - ev.py_cf).mean()),
            "_e_cf": e_cf, "_e_iv": e_iv, "_dcf": dcf, "_div": div,
        }
        per_seed.append(row)
        if sd == models[0]["seed"]:
            z_cf0, z_iv0 = row, None
    agg = _aggregate(per_seed)
    subsets = _subset_breakdown(per_seed[0], ev)
    gate = _gate_t2(agg, subsets)
    res = {"gate": "T2", "pass": gate["pass"], "verdict": gate["verdict"],
           "reasons": gate["reasons"], "aggregate": agg, "subsets": subsets,
           "per_seed": [{k: v for k, v in p.items() if not k.startswith("_")}
                        for p in per_seed],
           "oracle_frame_ok": ck.get("ae_train_git_commit", ""),
           "n_eval": int(len(Zt)), "iv_noise_seeds": n_iv}
    (OUT / "counterfactual_metrics.json").write_text(json.dumps(res, indent=2, default=_jd))
    _plots(per_seed[0], ev, dir_cf_true, OUT)
    print(f"[T2] {gate['verdict']}  " + "  ".join(gate["reasons"]))
    return res


def _aggregate(per_seed):
    keys = [k for k in per_seed[0] if not k.startswith("_") and k != "seed"]
    return {k: {"mean": float(np.mean([p[k] for p in per_seed])),
                "std": float(np.std([p[k] for p in per_seed]))} for k in keys}


def _subset_breakdown(s0, ev):
    e_cf, e_iv, dcf = s0["_e_cf"], s0["_e_iv"], s0["_dcf"]
    dir_cf_true = np.sign(ev.py_cf - ev.py_k).astype(int)
    masks = {"all": np.ones(len(e_cf), bool), "non_boundary": ~ev.boundary,
             "free": ev.stuck == 0, "stuck": ev.stuck == 1,
             "identifiable": ev.identifiable & (ev.stuck >= 0)}
    rows = []
    for name, msk in masks.items():
        if msk.sum() < 5:
            rows.append({"subset": name, "n": int(msk.sum())}); continue
        rows.append({"subset": name, "n": int(msk.sum()),
                     "cf_mse": round(float(e_cf[msk].mean()), 4),
                     "iv_mse": round(float(e_iv[msk].mean()), 4),
                     "paired_iv_minus_cf": round(float((e_iv - e_cf)[msk].mean()), 4),
                     "frac_cf_better": round(float((e_cf < e_iv)[msk].mean()), 3),
                     "cf_dir_acc": round(float((dcf[msk] == dir_cf_true[msk]).mean()), 3)})
    return rows


def _gate_t2(agg, subsets):
    a = lambda k: agg[k]["mean"]
    latent_cf_better = a("cf_mse") < a("iv_mse") and a("frac_cf_better") > 0.5
    # not only one small subset: paired advantage positive on >=2 subsets w/ n>=15
    pos_subsets = [s for s in subsets if s.get("n", 0) >= 15
                   and s.get("paired_iv_minus_cf", 0) > 0]
    broad = len(pos_subsets) >= 2
    physical = a("cf_dir_acc") > a("iv_dir_acc") or a("cf_pos_err") < a("iv_pos_err")
    not_copy = a("cf_mse") <= a("copyfac_mse") + 0.02   # CF not merely worse than copying
    stable = agg["paired_iv_minus_cf"]["std"] < abs(a("paired_iv_minus_cf")) + 1e-6
    reasons = [f"latent_CF<IV={latent_cf_better}(cf {a('cf_mse'):.3f} vs iv {a('iv_mse'):.3f}, "
               f"frac {a('frac_cf_better'):.2f})",
               f"broad={broad}({len(pos_subsets)} subsets)",
               f"physical={physical}(dir cf {a('cf_dir_acc'):.2f}/iv {a('iv_dir_acc'):.2f}, "
               f"pos cf {a('cf_pos_err'):.1f}/iv {a('iv_pos_err'):.1f})",
               f"not_copy={not_copy}(copyfac {a('copyfac_mse'):.3f})"]
    latent_ok = latent_cf_better and broad and not_copy
    if latent_ok and physical:
        verdict, passed = "PASS", True
    elif latent_ok and not physical:
        verdict, passed = "MIXED (latent only)", False
    else:
        verdict, passed = "FAIL", False
    return {"pass": passed, "verdict": verdict, "reasons": reasons}


def _plots(s0, ev, dir_cf_true, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from itertools import product
    e_cf, e_iv = s0["_e_cf"], s0["_e_iv"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    # paired scatter
    lim = [0, max(e_cf.max(), e_iv.max())]
    ax[0].scatter(e_iv, e_cf, s=12, alpha=0.5)
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("IV latent error"); ax[0].set_ylabel("CF latent error")
    ax[0].set_title(f"T2 paired CF vs IV (below diag = CF better, "
                    f"{(e_cf<e_iv).mean()*100:.0f}%)")
    # per-subset paired bars
    masks = {"all": np.ones(len(e_cf), bool), "free": ev.stuck == 0, "stuck": ev.stuck == 1,
             "non_bnd": ~ev.boundary}
    names = list(masks); vals = [float((e_iv - e_cf)[masks[n]].mean()) for n in names]
    ax[1].bar(names, vals, color=["tab:blue" if v > 0 else "tab:red" for v in vals])
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set_ylabel("mean(IV err − CF err)"); ax[1].set_title("T2 CF advantage by subset")
    # CF direction confusion
    labels = [-1, 0, 1]; cm = np.zeros((3, 3), int)
    for t, p in zip(dir_cf_true, s0["_dcf"]):
        cm[labels.index(t), labels.index(p)] += 1
    ax[2].imshow(cm, cmap="Blues")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["down", "stay", "up"])
    ax[2].set_yticks(range(3)); ax[2].set_yticklabels(["down", "stay", "up"])
    for i, j in product(range(3), range(3)):
        ax[2].text(j, i, cm[i, j], ha="center", va="center")
    ax[2].set_xlabel("pred"); ax[2].set_ylabel("true CF")
    ax[2].set_title(f"T2 CF direction acc={s0['cf_dir_acc']:.2f}")
    fig.tight_layout(); fig.savefig(out / "cf_vs_iv.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    evaluate()
