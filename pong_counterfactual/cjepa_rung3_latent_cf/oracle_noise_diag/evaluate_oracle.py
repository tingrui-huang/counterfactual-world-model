"""Oracle-noise factual (O1) and counterfactual (O2) evaluation.

Uses the TRUE sticky label as the noise. O1 = factual next-latent prediction vs
copy/no-noise/wrong-noise/wrong-action/shuffled controls. O2 (only if O1) = oracle-noise
CF vs IV / copy-factual-next / copy-current / wrong-noise / shuffled-noise, in latent AND
physical (frozen position probe) metrics. Writes metrics, plots, CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf import transition as TX
from pong_counterfactual.cjepa_rung3_latent_cf.evaluate_factual import readout_py, _per_mse
from pong_counterfactual.cjepa_rung3_latent_cf.oracle_noise_diag import oracle_noise as ON

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "main"
DIRLAB = [-1, 0, 1]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_run(cfg=None):
    cfg = cfg or ON.load_config()
    ck = torch.load(OUT / "oracle_ckpt.pt", map_location="cpu", weights_only=False)
    norm = LA.Normalizer(ck["norm_mu"], ck["norm_sd"])
    pr = ck["probe"]; probe = LA.PositionProbe(pr["coef"], pr["intercept"], pr["mu"], pr["sd"])

    def build(sd_list):
        out = []
        for md in sd_list:
            m = TX.LatentTransition(192, 8, cfg["K"], cfg["hidden"], cfg["depth"])
            m.load_state_dict(md["state_dict"]); m.eval()
            out.append(m)
        return out
    return build(ck["models"]), build(ck["no_noise"]), norm, probe, ck["prior"], cfg, ck


@torch.no_grad()
def fwd(m, Zt, A, u):
    return m(TX._to_t(Zt), TX._to_t(A), TX._to_t(np.asarray(u), long=True)).numpy()


def _dir(probe, norm, z_pred, z_ref):
    return np.sign(readout_py(probe, norm, z_pred) - readout_py(probe, norm, z_ref)).astype(int)


# =============================================================== O1 factual
def evaluate_o1(cfg=None):
    OUT.mkdir(parents=True, exist_ok=True)
    models, no_noise, norm, probe, prior, cfg, ck = load_run(cfg)
    ev = LA.build_eval_tuples(cfg)
    u = ev.stuck.astype(np.int64)                               # ORACLE noise
    Zt, Zt1 = norm.fwd(ev.z_t), norm.fwd(ev.z_t1)
    A = LA.action_feats(ev.a_int, ev.a_prev)
    Aw = LA.action_feats(ev.a_prime, ev.a_prev)                # wrong (swapped) action
    dir_true = np.sign(ev.py_k1 - ev.py_k).astype(int)
    maj = np.bincount(dir_true + 1).argmax() - 1
    rng = np.random.default_rng(0); perm = rng.permutation(len(Zt))
    rows = []
    for mi, (m, mn) in enumerate(zip(models, no_noise)):
        p_or = fwd(m, Zt, A, u)
        e = {"oracle": _per_mse(p_or, Zt1),
             "copy_current": _per_mse(Zt, Zt1),
             "no_noise": _per_mse(fwd(mn, Zt, A, np.zeros_like(u)), Zt1),
             "wrong_noise": _per_mse(fwd(m, Zt, A, 1 - u), Zt1),
             "wrong_action": _per_mse(fwd(m, Zt, Aw, u), Zt1),
             "shuffled_next": _per_mse(p_or, Zt1[perm])}
        dpred = _dir(probe, norm, p_or, Zt)
        rows.append({"seed": mi, **{f"mse_{k}": float(v.mean()) for k, v in e.items()},
                     "dir_acc": float((dpred == dir_true).mean()),
                     "dir_majority": float((dir_true == maj).mean()),
                     "bal_acc": float(balanced_accuracy_score(dir_true, dpred)),
                     "pos_err": float(np.abs(readout_py(probe, norm, p_or) - ev.py_k1).mean()),
                     "_e": e, "_dpred": dpred})
    agg = _agg(rows)
    subs = _subsets_o1(models[0], norm, probe, ev, Zt, Zt1, A, u, dir_true)
    g = _gate_o1(agg)
    res = {"gate": "O1", "pass": g["pass"], "reasons": g["reasons"], "aggregate": agg,
           "subsets": subs, "per_seed": [{k: v for k, v in r.items()
                                          if not k.startswith("_")} for r in rows]}
    (OUT / "oracle_factual_metrics.json").write_text(json.dumps(res, indent=2, default=_jd))
    print(f"[O1] pass={g['pass']}  " + "  ".join(g["reasons"]))
    return res


def _subsets_o1(m, norm, probe, ev, Zt, Zt1, A, u, dir_true):
    p = fwd(m, Zt, A, u); dpred = _dir(probe, norm, p, Zt); e = _per_mse(p, Zt1)
    ec = _per_mse(Zt, Zt1)
    masks = _masks(ev)
    out = []
    for name, msk in masks.items():
        if msk.sum() < 5:
            out.append({"subset": name, "n": int(msk.sum())}); continue
        out.append({"subset": name, "n": int(msk.sum()),
                    "dir_acc": round(float((dpred[msk] == dir_true[msk]).mean()), 3),
                    "mse_oracle": round(float(e[msk].mean()), 4),
                    "mse_copy_current": round(float(ec[msk].mean()), 4)})
    return out


def _gate_o1(a):
    f = lambda k: a[k]["mean"]
    # task's literal criterion: correct noise improves over BOTH no-noise and wrong-noise
    # (any positive margin). The MAGNITUDE of the no-noise gap is reported as a warning —
    # a tiny gap already foreshadows a latent/forward (not noise) bottleneck.
    c_noise = (f("mse_oracle") < f("mse_no_noise") - 1e-3
               and f("mse_oracle") < f("mse_wrong_noise") - 1e-3)
    c_act = f("mse_oracle") < f("mse_wrong_action") - 1e-4
    c_copy = f("mse_oracle") < 0.9 * f("mse_copy_current")
    c_dir = f("dir_acc") > f("dir_majority") + 0.03
    gap = f("mse_no_noise") - f("mse_oracle")
    warn = "MARGINAL noise benefit" if gap < 0.02 else ""
    reasons = [f"noise>{c_noise}(or {f('mse_oracle'):.3f} vs nonoise {f('mse_no_noise'):.3f} "
               f"[gap {gap:+.3f} {warn}] / wrong {f('mse_wrong_noise'):.3f})",
               f"action={c_act}(wrong {f('mse_wrong_action'):.3f})",
               f"copy={c_copy}(cur {f('mse_copy_current'):.3f})",
               f"dir={c_dir}({f('dir_acc'):.3f}>maj {f('dir_majority'):.3f})"]
    return {"pass": bool(c_noise and c_act and c_copy and c_dir),
            "reasons": reasons, "marginal_noise_benefit": bool(gap < 0.02)}


# =============================================================== O2 counterfactual
def evaluate_o2(cfg=None):
    o1 = json.loads((OUT / "oracle_factual_metrics.json").read_text())
    if not o1["pass"]:
        (OUT / "oracle_cf_metrics.json").write_text(json.dumps(
            {"gate": "O2", "status": "NOT RUN", "reason": "O1 failed"}, indent=2))
        print("[O2] NOT RUN — O1 failed."); return {"status": "NOT RUN", "verdict": "NOT RUN"}
    models, no_noise, norm, probe, prior, cfg, ck = load_run(cfg)
    ev = LA.build_eval_tuples(cfg)
    u = ev.stuck.astype(np.int64)
    Zt, Zt1, Zcf = norm.fwd(ev.z_t), norm.fwd(ev.z_t1), norm.fwd(ev.z_cf_oracle)
    Af = LA.action_feats(ev.a_int, ev.a_prev)
    Ac = LA.action_feats(ev.a_prime, ev.a_prev)
    dir_cf = np.sign(ev.py_cf - ev.py_k).astype(int)
    maj = np.bincount(dir_cf + 1).argmax() - 1
    rng = np.random.default_rng(1); perm = rng.permutation(len(Zt))
    rows = []
    for mi, m in enumerate(models):
        p_cf = fwd(m, Zt, Ac, u)                                # oracle-noise CF
        e_cf = _per_mse(p_cf, Zcf)
        # IV: fresh u~prior
        e_iv_d = []
        r2 = np.random.default_rng(100 + mi); z_iv_last = None
        for _ in range(cfg["iv_noise_seeds"]):
            uiv = r2.choice(len(prior), len(Zt), p=prior)
            z_iv_last = fwd(m, Zt, Ac, uiv); e_iv_d.append(_per_mse(z_iv_last, Zcf))
        e_iv = np.mean(e_iv_d, 0)
        methods = {
            "oracle_cf": (p_cf, e_cf),
            "iv": (z_iv_last, e_iv),
            "copy_fac_next": (Zt1, _per_mse(Zt1, Zcf)),
            "copy_current": (Zt, _per_mse(Zt, Zcf)),
            "wrong_noise": (fwd(m, Zt, Ac, 1 - u), None),
            "shuffled_noise": (fwd(m, Zt, Ac, u[perm]), None),
            "wrong_action": (fwd(m, Zt, Af, u), None),      # un-swapped action
        }
        row = {"seed": mi}
        for name, (z, e) in methods.items():
            if e is None:
                e = _per_mse(z, Zcf)
            dpred = _dir(probe, norm, z, Zt)
            row[f"mse_{name}"] = float(e.mean())
            row[f"dir_{name}"] = float((dpred == dir_cf).mean())
            row[f"bal_{name}"] = float(balanced_accuracy_score(dir_cf, dpred))
            row[f"pos_{name}"] = float(np.abs(readout_py(probe, norm, z) - ev.py_cf).mean())
            row[f"betteriv_{name}"] = float((e < e_iv).mean())
            if name in ("oracle_cf", "iv", "copy_fac_next"):
                row[f"_dpred_{name}"] = dpred
            if name == "oracle_cf":
                row["_e_cf"] = e; row["paired_iv_minus_cf"] = float((e_iv - e_cf).mean())
            if name == "copy_fac_next":
                row["_e_copyfac"] = e
        row["_e_iv"] = e_iv
        rows.append(row)
    agg = _agg(rows)
    subs = _subsets_o2(rows[0], ev)
    copyb = _copy_breakdown(rows[0], ev)
    g = _gate_o2(agg, subs, copyb, rows[0], ev, float(maj))
    res = {"gate": "O2", "pass": g["pass"], "verdict": g["verdict"], "reasons": g["reasons"],
           "aggregate": agg, "subsets": subs, "copy_breakdown": copyb, "majority": float(maj),
           "per_seed": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
    (OUT / "oracle_cf_metrics.json").write_text(json.dumps(res, indent=2, default=_jd))
    _write_csv(agg)
    _plots(rows[0], ev, dir_cf, agg, copyb)
    _gallery(ev, rows[0], probe, norm, Zt, Zcf)
    print(f"[O2] {g['verdict']}  " + "  ".join(g["reasons"]))
    return res


def _masks(ev):
    return {"all": np.ones(len(ev.stuck), bool), "stuck": ev.stuck == 1, "free": ev.stuck == 0,
            "boundary": ev.boundary, "non_boundary": ~ev.boundary,
            "identifiable": (~ev.boundary), "ambiguous": ev.boundary}


def _subsets_o2(r, ev):
    e_cf, e_iv = r["_e_cf"], r["_e_iv"]; dcf = r["_dpred_oracle_cf"]
    dir_cf = np.sign(ev.py_cf - ev.py_k).astype(int)
    out = []
    for name, msk in _masks(ev).items():
        if msk.sum() < 5:
            out.append({"subset": name, "n": int(msk.sum())}); continue
        out.append({"subset": name, "n": int(msk.sum()),
                    "cf_mse": round(float(e_cf[msk].mean()), 4),
                    "iv_mse": round(float(e_iv[msk].mean()), 4),
                    "paired_iv_minus_cf": round(float((e_iv - e_cf)[msk].mean()), 4),
                    "frac_cf_better_iv": round(float((e_cf < e_iv)[msk].mean()), 3),
                    "cf_dir_acc": round(float((dcf[msk] == dir_cf[msk]).mean()), 3)})
    return out


def _copy_breakdown(r, ev):
    e_cf, e_copyfac = r["_e_cf"], r["_e_copyfac"]
    out = []
    for name, msk in _masks(ev).items():
        if msk.sum() < 5 or name in ("identifiable", "ambiguous"):
            continue
        out.append({"subset": name, "n": int(msk.sum()),
                    "cf_mse": round(float(e_cf[msk].mean()), 4),
                    "copyfac_mse": round(float(e_copyfac[msk].mean()), 4)})
    return out


def _gate_o2(a, subs, copyb, r, ev, maj):
    f = lambda k: a[k]["mean"]
    cf_beats_iv = f("mse_oracle_cf") < f("mse_iv") and f("betteriv_oracle_cf") > 0.5
    # copy-factual-next dominates STUCK tuples by construction (it holds the true ball).
    # The relevant test is the FREE subset (where the intervention actually branches).
    free_cb = next((s for s in copyb if s["subset"] == "free"), None)
    cf_vs_copyfac = bool(free_cb and free_cb["cf_mse"] <= free_cb["copyfac_mse"] + 0.02)
    free_sub = next((s for s in subs if s["subset"] == "free"), None)
    # physical: direction on the free subset (aggregate direction is confounded by stuck)
    physical = bool(free_sub and free_sub["cf_dir_acc"] > 0.55) or \
        (f("dir_oracle_cf") > f("dir_iv") + 0.03)
    wrong_removes = f("mse_wrong_noise") > f("mse_oracle_cf") + 1e-3 and \
        f("mse_shuffled_noise") > f("mse_oracle_cf") + 1e-3
    dcf = r["_dpred_oracle_cf"]; stay_pred = int((dcf[ev.stuck == 1] == 0).sum())
    freecf = free_sub["cf_dir_acc"] if free_sub else float("nan")
    reasons = [f"cf<iv={cf_beats_iv}(cf {f('mse_oracle_cf'):.3f}/iv {f('mse_iv'):.3f}, "
               f"frac {f('betteriv_oracle_cf'):.2f})",
               f"vs_copyfac_FREE={cf_vs_copyfac}("
               f"{'cf %.3f/copy %.3f' % (free_cb['cf_mse'], free_cb['copyfac_mse']) if free_cb else 'na'})",
               f"physical={physical}(free dir {freecf:.2f}; agg dir cf {f('dir_oracle_cf'):.2f}"
               f"/iv {f('dir_iv'):.2f})",
               f"wrong_removes={wrong_removes}(wrong {f('mse_wrong_noise'):.3f}/"
               f"shuf {f('mse_shuffled_noise'):.3f})", f"stay_pred_on_stuck={stay_pred}"]
    strong = cf_beats_iv and cf_vs_copyfac and physical and wrong_removes
    # decisive negatives for the noise-value question:
    #  - noise_spurious: scrambling the noise (wrong/shuffled) does NOT degrade the CF, so
    #    the tiny cf<iv margin is not attributable to the correct exogenous realization;
    #  - physical_at_chance: direction readout ~ chance and balanced acc ~ 1/3.
    noise_spurious = not wrong_removes
    physical_at_chance = f("dir_oracle_cf") < maj + 0.05 and f("bal_oracle_cf") < 0.40
    copy_dominates = f("mse_oracle_cf") > f("mse_copy_fac_next") + 0.05
    if strong:
        verdict, passed = "ORACLE-NOISE SUCCESS", True
    elif (not cf_beats_iv) or noise_spurious or (physical_at_chance and copy_dominates):
        verdict, passed = "ORACLE-NOISE FAILURE", False
    else:
        verdict, passed = "MIXED / INCONCLUSIVE", False
    reasons.append(f"noise_spurious={noise_spurious} physical_at_chance={physical_at_chance} "
                   f"copy_dominates={copy_dominates}")
    return {"pass": passed, "verdict": verdict, "reasons": reasons}


def _agg(rows):
    keys = [k for k in rows[0] if not k.startswith("_") and k != "seed"]
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows]))} for k in keys}


def _write_csv(agg):
    f = lambda k: agg[k]["mean"]
    methods = [("Oracle-noise CF", "oracle_cf"), ("IV", "iv"),
               ("Copy factual next", "copy_fac_next"), ("Copy current", "copy_current"),
               ("Wrong noise", "wrong_noise"), ("Shuffled noise", "shuffled_noise")]
    with open(OUT / "oracle_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "latent_mse", "dir_acc", "bal_acc", "pos_err", "beats_iv_frac"])
        for lab, k in methods:
            w.writerow([lab, f"{f('mse_'+k):.4f}", f"{f('dir_'+k):.3f}", f"{f('bal_'+k):.3f}",
                        f"{f('pos_'+k):.2f}", f"{f('betteriv_'+k):.3f}"])


def _plots(r, ev, dir_cf, agg, copyb):
    plt = _plt(); from itertools import product
    e_cf, e_iv = r["_e_cf"], r["_e_iv"]
    # fig1: paired + subset advantage + noise usefulness
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    lim = [0, max(e_cf.max(), e_iv.max())]
    ax[0].scatter(e_iv, e_cf, s=12, alpha=0.5); ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("IV latent err"); ax[0].set_ylabel("Oracle-CF latent err")
    ax[0].set_title(f"O2 paired (CF<IV {(e_cf<e_iv).mean()*100:.0f}%)")
    names = [s["subset"] for s in copyb]
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [s["cf_mse"] for s in copyb], 0.4, label="oracle-CF")
    ax[1].bar(x + 0.2, [(s["copyfac_mse"] or 0) for s in copyb], 0.4, label="copy-fac-next")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("latent MSE"); ax[1].set_title("O2 CF vs copy-fac-next by subset"); ax[1].legend(fontsize=8)
    ax[1].tick_params(axis='x', rotation=20)
    f = lambda k: agg[k]["mean"]
    nu = ["oracle_cf", "wrong_noise", "shuffled_noise", "iv"]
    ax[2].bar([n.replace("_", "\n") for n in nu], [f("mse_" + n) for n in nu], color="tab:purple")
    ax[2].set_ylabel("CF latent MSE"); ax[2].set_title("O2 noise usefulness")
    fig.tight_layout(); fig.savefig(OUT / "oracle_cf_overview.png", dpi=110); plt.close(fig)
    # fig2: confusion matrices oracle-CF / IV / copy-fac-next
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for j, key in enumerate(["oracle_cf", "iv", "copy_fac_next"]):
        cm = confusion_matrix(dir_cf, r[f"_dpred_{key}"], labels=DIRLAB)
        ax[j].imshow(cm, cmap="Blues")
        ax[j].set_xticks(range(3)); ax[j].set_xticklabels(["down", "stay", "up"])
        ax[j].set_yticks(range(3)); ax[j].set_yticklabels(["down", "stay", "up"])
        for i, c in product(range(3), range(3)):
            ax[j].text(c, i, cm[i, c], ha="center", va="center")
        ax[j].set_title(key); ax[j].set_xlabel("pred")
        if j == 0:
            ax[j].set_ylabel("true CF")
    fig.suptitle("O2 direction confusion (note the 'stay' column)")
    fig.tight_layout(); fig.savefig(OUT / "oracle_cf_confusion.png", dpi=110); plt.close(fig)


def _gallery(ev, r, probe, norm, Zt, Zcf):
    plt = _plt()
    b = LA.CM.load_eval_bundle(LA._p(ON.load_config()["data_dir"]))
    e_cf = r["_e_cf"]
    free = np.where(ev.stuck == 0)[0]; stuck = np.where(ev.stuck == 1)[0]
    pick = []
    if len(free):
        pick += [("free-best", free[np.argsort(e_cf[free])[0]]),
                 ("free-worst", free[np.argsort(-e_cf[free])[0]])]
    if len(stuck):
        pick += [("stuck-best", stuck[np.argsort(e_cf[stuck])[0]]),
                 ("stuck-worst", stuck[np.argsort(-e_cf[stuck])[0]])]
    fig, ax = plt.subplots(3, len(pick), figsize=(2.4 * len(pick), 5.2))
    for j, (lab, i) in enumerate(pick):
        for rrow, (img, t) in enumerate([(b.cur_frame[i], "factual t"),
                                         (b.fac_frame[i], "factual t+1"),
                                         (b.cf_frame[i], "CF oracle t+1")]):
            ax[rrow, j].imshow(img, cmap="gray"); ax[rrow, j].set_xticks([]); ax[rrow, j].set_yticks([])
            if rrow == 0:
                ax[rrow, j].set_title(f"{lab}\na {int(ev.a_int[i])}->{int(ev.a_prime[i])} "
                                      f"u={int(ev.stuck[i])}", fontsize=7)
            if j == 0:
                ax[rrow, j].set_ylabel(t, fontsize=8)
    fig.suptitle("O2 examples (factual / factual-next / CF-oracle)")
    fig.tight_layout(); fig.savefig(OUT / "oracle_cf_gallery.png", dpi=110); plt.close(fig)


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    evaluate_o1(); evaluate_o2()
