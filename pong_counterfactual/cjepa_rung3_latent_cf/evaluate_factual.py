"""Factual transition qualification (Gate T1).

On the fixed 250-tuple eval pool: abduct u, predict the factual next latent, and compare
to copy / no-noise(marginal) / wrong-action / shuffled-next controls, plus a frozen
position-probe direction readout. Also report abduced-u vs oracle-stuck agreement.
Run: python -m pong_counterfactual.cjepa_rung3_latent_cf.evaluate_factual
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf import transition as TX

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "main"


# ------------------------------------------------------------------ shared helpers
def load_run(cfg=None):
    cfg = cfg or LA.load_config()
    ck = torch.load(OUT / "transition_ckpt.pt", map_location="cpu", weights_only=False)
    norm = LA.Normalizer(ck["norm_mu"], ck["norm_sd"])
    pr = ck["probe"]
    probe = LA.PositionProbe(pr["coef"], pr["intercept"], pr["mu"], pr["sd"])
    models = []
    for md in ck["models"]:
        m = TX.LatentTransition(192, 8, cfg["K"], cfg["hidden"], cfg["depth"])
        m.load_state_dict(md["state_dict"]); m.eval()
        prior = np.bincount(md["u_train"], minlength=cfg["K"]).astype(float)
        prior /= prior.sum()
        models.append({"model": m, "prior": prior, "seed": md["seed"]})
    return models, norm, probe, cfg, ck


@torch.no_grad()
def marginal_predict(model, Zt, A, prior):
    """Prior-weighted forward prediction over u (the NO-abduction / no-noise baseline)."""
    preds = model.predict_all_u(TX._to_t(Zt), TX._to_t(A))       # (N,K,z)
    w = torch.tensor(prior, dtype=torch.float32)[None, :, None]
    return (preds * w).sum(1).numpy()


@torch.no_grad()
def abduct_predict(model, Zt, A, Zt1_obs):
    """Abduct u from the observed next latent, then predict with that u (factual)."""
    u_hat, _ = model.abduct(TX._to_t(Zt), TX._to_t(A), TX._to_t(Zt1_obs))
    pred = model(TX._to_t(Zt), TX._to_t(A), u_hat)
    return pred.numpy(), u_hat.numpy()


def readout_py(probe, norm, z_norm):
    """normalized latent -> player_y (screen px) via the frozen position probe."""
    return probe.predict(norm.inv(z_norm))


def _mse(a, b):
    return float(((a - b) ** 2).mean())


def _per_mse(a, b):
    return ((a - b) ** 2).mean(1)


# ------------------------------------------------------------------ T1
def evaluate(cfg=None):
    OUT.mkdir(parents=True, exist_ok=True)
    models, norm, probe, cfg, ck = load_run(cfg)
    ev = LA.build_eval_tuples(cfg)
    Zt, Zt1 = norm.fwd(ev.z_t), norm.fwd(ev.z_t1)
    A = LA.action_feats(ev.a_int, ev.a_prev)
    A_wrong = LA.action_feats(ev.a_prime, ev.a_prev)          # wrong (swapped) action
    dir_true = np.sign(ev.py_k1 - ev.py_k).astype(int)
    maj = np.bincount((dir_true + 1)).argmax() - 1
    rng = np.random.default_rng(0)

    per_seed = []
    for mm in models:
        m, prior = mm["model"], mm["prior"]
        pred_ab, u_hat = abduct_predict(m, Zt, A, Zt1)
        pred_marg = marginal_predict(m, Zt, A, prior)
        pred_wrong, _ = abduct_predict(m, Zt, A_wrong, Zt1)  # abduct+predict w/ wrong action
        copy_pred = Zt                                       # copy baseline (norm space)
        perm = rng.permutation(len(Zt))
        # latent errors (normalized space)
        e_ab = _per_mse(pred_ab, Zt1)
        e_marg = _per_mse(pred_marg, Zt1)
        e_copy = _per_mse(copy_pred, Zt1)
        e_wrong = _per_mse(pred_wrong, Zt1)
        e_shuf = _per_mse(pred_marg, Zt1[perm])             # correct-pair vs shuffled next
        # direction readout from the abduction-predicted next latent
        py_pred = readout_py(probe, norm, pred_ab)
        py_t = readout_py(probe, norm, Zt)
        dir_pred = np.sign(py_pred - py_t).astype(int)
        dir_acc = float((dir_pred == dir_true).mean())
        pos_err = float(np.abs(py_pred - ev.py_k1).mean())
        pos_err_copy = float(np.abs(py_t - ev.py_k1).mean())
        # abduced-u vs oracle stuck (best mapping accuracy), identifiable subset
        idf = ev.identifiable & (ev.stuck >= 0)
        u_acc = _best_map_acc(u_hat[idf], ev.stuck[idf], cfg["K"])
        per_seed.append({
            "seed": mm["seed"],
            "mse_abduct": float(e_ab.mean()), "mse_marginal": float(e_marg.mean()),
            "mse_copy": float(e_copy.mean()), "mse_wrong_action": float(e_wrong.mean()),
            "mse_shuffled_next": float(e_shuf.mean()),
            "nmse_vs_copy": float(e_ab.mean() / e_copy.mean()),
            "dir_acc": dir_acc, "dir_majority": float((dir_true == maj).mean()),
            "pos_err_px": pos_err, "pos_err_copy_px": pos_err_copy,
            "abduced_u_vs_stuck_acc": u_acc,
            "u_sizes_eval": np.bincount(u_hat, minlength=cfg["K"]).tolist(),
            "_u_hat": u_hat, "_dir_pred": dir_pred,
        })
    agg = _aggregate(per_seed)
    # subset breakdown (seed 0 model) on direction + latent error
    subsets = _subsets(models[0], norm, probe, ev, Zt, Zt1, A, dir_true, cfg)

    gate = _gate_t1(agg)
    res = {"gate": "T1", "pass": gate["pass"], "reasons": gate["reasons"],
           "aggregate": agg, "per_seed": [{k: v for k, v in p.items()
                                           if not k.startswith("_")} for p in per_seed],
           "subsets": subsets, "majority_dir": float(maj),
           "n_eval": int(len(Zt))}
    (OUT / "factual_metrics.json").write_text(json.dumps(res, indent=2, default=_jd))
    _plot_factual(agg, per_seed[0], dir_true, OUT)
    print(f"[T1] pass={gate['pass']}  " + "  ".join(gate["reasons"]))
    return res


def _best_map_acc(u, stuck, K):
    """Best label-permutation accuracy of abduced u against binary oracle stuck."""
    if len(u) == 0:
        return None
    best = 0.0
    # K=2: try both mappings mode->stuck
    import itertools
    for perm in itertools.product([0, 1], repeat=K):
        pred = np.array([perm[ui] for ui in u])
        best = max(best, float((pred == stuck).mean()))
    return best


def _aggregate(per_seed):
    keys = [k for k in per_seed[0] if not k.startswith("_") and k != "seed"
            and not isinstance(per_seed[0][k], list)]
    out = {}
    for k in keys:
        vals = np.array([p[k] for p in per_seed if p[k] is not None], float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def _subsets(mm, norm, probe, ev, Zt, Zt1, A, dir_true, cfg):
    m, prior = mm["model"], mm["prior"]
    pred_ab, _ = abduct_predict(m, Zt, A, Zt1)
    py_pred = readout_py(probe, norm, pred_ab); py_t = readout_py(probe, norm, Zt)
    dir_pred = np.sign(py_pred - py_t).astype(int)
    e_ab = _per_mse(pred_ab, Zt1); e_copy = _per_mse(Zt, Zt1)
    masks = {"all": np.ones(len(Zt), bool), "non_boundary": ~ev.boundary,
             "free": ev.stuck == 0, "stuck": ev.stuck == 1,
             "identifiable": ev.identifiable & (ev.stuck >= 0)}
    rows = []
    for name, msk in masks.items():
        if msk.sum() < 5:
            rows.append({"subset": name, "n": int(msk.sum())}); continue
        rows.append({"subset": name, "n": int(msk.sum()),
                     "dir_acc": round(float((dir_pred[msk] == dir_true[msk]).mean()), 3),
                     "mse_abduct": round(float(e_ab[msk].mean()), 4),
                     "mse_copy": round(float(e_copy[msk].mean()), 4)})
    return rows


def _gate_t1(agg):
    reasons = []
    a = lambda k: agg[k]["mean"]
    c_copy = a("mse_abduct") < 0.9 * a("mse_copy")
    c_act = a("mse_abduct") < a("mse_wrong_action") - 1e-4
    c_shuf = a("mse_marginal") < 0.9 * a("mse_shuffled_next")
    c_noise = a("mse_abduct") < 0.9 * a("mse_marginal")
    c_dir = a("dir_acc") > a("dir_majority") + 0.03
    reasons = [f"beat_copy={c_copy}(ab {a('mse_abduct'):.3f}<copy {a('mse_copy'):.3f})",
               f"action={c_act}(wrong {a('mse_wrong_action'):.3f})",
               f"shuffle={c_shuf}(shuf {a('mse_shuffled_next'):.3f})",
               f"noise>{c_noise}(marg {a('mse_marginal'):.3f})",
               f"dir={c_dir}({a('dir_acc'):.3f}>maj {a('dir_majority'):.3f})"]
    return {"pass": bool(c_copy and c_act and c_shuf and c_noise and c_dir),
            "reasons": reasons}


def _plot_factual(agg, s0, dir_true, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labs = ["abduct", "marginal\n(no-noise)", "copy", "wrong\naction", "shuffled\nnext"]
    keys = ["mse_abduct", "mse_marginal", "mse_copy", "mse_wrong_action", "mse_shuffled_next"]
    ax[0].bar(labs, [agg[k]["mean"] for k in keys],
              yerr=[agg[k]["std"] for k in keys], color="tab:blue")
    ax[0].set_ylabel("normalized latent MSE"); ax[0].set_title("T1 factual: model vs baselines")
    from itertools import product
    dp = s0["_dir_pred"]
    labels = [-1, 0, 1]
    cm = np.zeros((3, 3), int)
    for t, p in zip(dir_true, dp):
        cm[labels.index(t), labels.index(p)] += 1
    ax[1].imshow(cm, cmap="Blues")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["down", "stay", "up"])
    ax[1].set_yticks(range(3)); ax[1].set_yticklabels(["down", "stay", "up"])
    for i, j in product(range(3), range(3)):
        ax[1].text(j, i, cm[i, j], ha="center", va="center")
    ax[1].set_xlabel("pred"); ax[1].set_ylabel("true")
    ax[1].set_title(f"T1 direction readout (seed {s0['seed']}) acc={s0['dir_acc']:.2f}")
    fig.tight_layout(); fig.savefig(out / "factual_direction.png", dpi=110); plt.close(fig)


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    evaluate()
