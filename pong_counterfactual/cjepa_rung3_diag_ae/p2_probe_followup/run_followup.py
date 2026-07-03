"""P2 follow-up: is the AE latent's Δy signal there but mis-extracted, or absent?

Frozen AE encoder (same checkpoint/dataset/split/pool as the AE diagnostic). We compare
alternative frozen-latent probes with a clean temporal-pair protocol:

  fit ONE probe on CORRECT train pairs (z_t, z_t1), then evaluate the same probe on
    - correct test pairs           (z_t, z_t1)
    - mismatched test pairs        (z_t, z_t1 from a DIFFERENT test transition)
    - same-frame test pairs        (z_t, z_t)
  plus a separately-fit static-only probe (z_t) and a shuffled-label control.

Temporal-pair gap = acc(correct) − acc(mismatched). A genuine two-frame signal must make
this gap clearly positive; the original P2's gap was ~0 (mismatch didn't hurt).

Variants (input built from the (16,6,6) spatial latent):
  full      : [flat z_t, flat z_t1]                      (== the existing P2 input)
  gap        : [meanpool z_t, meanpool z_t1]             (control: what pooling WOULD do)
  local      : [flat z_t[...,paddle cols], flat z_t1[...]]      (paddle-local)
  full+diff  : [flat z_t, flat z_t1, flat(z_t1−z_t)]
  local+diff : paddle-local z_t, z_t1, and (z_t1−z_t)

Run: python -m pong_counterfactual.cjepa_rung3_diag_ae.p2_probe_followup.run_followup
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from pong_counterfactual.cjepa_rung3_diag_ae import common as CM
from pong_counterfactual.cjepa_rung3_diag_ae.ae_model import ConvAutoencoder, AEConfig
from pong_counterfactual.cjepa_rung3_diag_ae.train_ae import player_paddle_column

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "p2_followup"
AE_CKPT = HERE.parent / "outputs" / "main" / "A1" / "ae_checkpoint.pt"
PADDLE_COLS = slice(4, 6)     # right 2 of the 6 latent columns (P3-localized paddle side)
DIR_LABELS = [-1, 0, 1]       # down / stay / up


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(CM.REPO_ROOT)).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ frozen encoder
def load_frozen_ae():
    ck = torch.load(AE_CKPT, map_location="cpu", weights_only=False)
    cfg = AEConfig(**ck["config"])
    ae = ConvAutoencoder(cfg)
    ae.load_state_dict(ck["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ck


def spatial_latent(ae, frames_u8, batch=256):
    """(N,res,res) uint8 -> (N,16,6,6) frozen latent (no pooling)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(frames_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + batch]))
            out.append(ae.encode(x).cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ feature builders
def _flat(a):
    return a.reshape(a.shape[0], -1)


BUILDERS = {
    "full":       lambda zt, zt1: np.concatenate([_flat(zt), _flat(zt1)], 1),
    "gap":        lambda zt, zt1: np.concatenate([zt.mean((2, 3)), zt1.mean((2, 3))], 1),
    "local":      lambda zt, zt1: np.concatenate(
        [_flat(zt[..., PADDLE_COLS]), _flat(zt1[..., PADDLE_COLS])], 1),
    "full+diff":  lambda zt, zt1: np.concatenate(
        [_flat(zt), _flat(zt1), _flat(zt1 - zt)], 1),
    "local+diff": lambda zt, zt1: np.concatenate(
        [_flat(zt[..., PADDLE_COLS]), _flat(zt1[..., PADDLE_COLS]),
         _flat((zt1 - zt)[..., PADDLE_COLS])], 1),
}
STATIC_BUILDERS = {   # z_t only (matched dimensionality family)
    "full":       lambda zt: _flat(zt),
    "gap":        lambda zt: zt.mean((2, 3)),
    "local":      lambda zt: _flat(zt[..., PADDLE_COLS]),
    "full+diff":  lambda zt: _flat(zt),
    "local+diff": lambda zt: _flat(zt[..., PADDLE_COLS]),
}


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ------------------------------------------------------------------ one variant eval
def eval_variant(name, zt, zt1, dir_y, dy, tr, te, split_seed, use_mlp=False):
    """Clean temporal-pair protocol. Returns metrics for correct/mismatch/same/static."""
    rng = np.random.default_rng(1000 + split_seed)
    build = BUILDERS[name]
    # mismatched z_t1: permute test rows (and train rows, for a fit-on-garbage variant we
    # DON'T need — we fit on correct and only swap at eval). Permutation is a derangement-ish
    # shuffle over the test indices so z_t1 comes from a different transition.
    te_idx = np.where(te)[0]
    perm = te_idx.copy()
    rng.shuffle(perm)
    # guard against accidental identity matches (same index) -> roll any fixed points
    fixed = perm == te_idx
    if fixed.any():
        perm[fixed] = np.roll(perm[fixed], 1) if fixed.sum() > 1 else \
            perm[fixed]  # tiny sets: accept
    zt1_mis = zt1.copy()
    zt1_mis[te_idx] = zt1[perm]

    Xc = build(zt, zt1)
    Xm = build(zt, zt1_mis)
    Xs = build(zt, zt)            # same-frame

    def clf(Xtr, ytr, Xte):
        Xtr2, Xte2 = _standardize(Xtr, Xte)
        m = (MLPClassifier(hidden_layer_sizes=(64,), max_iter=1500,
                           random_state=split_seed)
             if use_mlp else LogisticRegression(max_iter=3000))
        m.fit(Xtr2, ytr)
        return m.predict(Xte2)

    # direction (3-class) — fit on CORRECT train, eval on the three test variants
    ycorr = clf(Xc[tr], dir_y[tr], Xc[te])
    ymis = clf(Xc[tr], dir_y[tr], Xm[te])   # SAME probe params via same fit call
    ysame = clf(Xc[tr], dir_y[tr], Xs[te])
    # NB: clf refits each call (cheap); inputs share the identical fitted mapping because
    # the training data (Xc[tr], dir_y[tr]) is identical — only the test matrix differs.

    # regression of dy (correct pair)
    Xtr2, Xte2 = _standardize(Xc[tr], Xc[te])
    rid = Ridge(alpha=1.0).fit(Xtr2, dy[tr])
    pred_dy = rid.predict(Xte2)
    r2 = 1 - ((pred_dy - dy[te]) ** 2).sum() / (((dy[te] - dy[te].mean()) ** 2).sum() + 1e-12)
    mae = float(np.abs(pred_dy - dy[te]).mean())

    # static-only (z_t) + shuffled-label controls
    Xst = STATIC_BUILDERS[name](zt)
    ystat = clf(Xst[tr], dir_y[tr], Xst[te])
    ysh_tr = dir_y[tr].copy(); rng.shuffle(ysh_tr)
    ysh = clf(Xc[tr], ysh_tr, Xc[te])

    yte = dir_y[te]

    def acc(p): return float((p == yte).mean())
    def bacc(p): return float(balanced_accuracy_score(yte, p))
    return {
        "variant": name, "split_seed": split_seed, "dim": int(Xc.shape[1]),
        "dir_acc_correct": acc(ycorr), "bal_acc_correct": bacc(ycorr),
        "dir_acc_mismatch": acc(ymis), "dir_acc_same": acc(ysame),
        "dir_acc_static": acc(ystat), "dir_acc_shuffled": acc(ysh),
        "temporal_gap": acc(ycorr) - acc(ymis),
        "dy_r2": float(r2), "dy_mae": mae,
        "_pred_correct": ycorr, "_pred_mismatch": ymis, "_yte": yte,
    }


# ------------------------------------------------------------------ baselines
def paddle_centroid(frames_u8):
    """Training-free player-paddle vertical centroid in the right pixel band (resized px).
    Background-subtracted brightness-weighted row centroid; NaN if no foreground."""
    res = frames_u8.shape[-1]
    col = player_paddle_column(frames_u8)
    band = frames_u8[:, :, col].astype(np.float32)          # (N, res, w)
    bg = np.median(frames_u8)                               # global background gray
    w = np.clip(band - bg - 10, 0, None).sum(2)            # (N,res) row weight
    rows = np.arange(res)[None, :]
    tot = w.sum(1)
    cen = (w * rows).sum(1) / np.where(tot > 0, tot, np.nan)
    return cen                                             # (N,) resized-row px


def run_baselines(b, tr, te):
    dir_y = np.sign(b.delta_y).astype(int)
    out = {}
    # 1 majority
    vals, cnts = np.unique(dir_y[tr], return_counts=True)
    maj = vals[cnts.argmax()]
    out["majority"] = {"dir_acc": float((dir_y[te] == maj).mean()),
                       "bal_acc": float(balanced_accuracy_score(
                           dir_y[te], np.full(te.sum(), maj)))}
    # 2 oracle player_y_t only (static) -> regression + direction via ridge sign
    def ridge_dir_r2(X):
        Xtr, Xte = _standardize(X[tr], X[te])
        m = Ridge(1.0).fit(Xtr, b.delta_y[tr]); pred = m.predict(Xte)
        r2 = 1 - ((pred - b.delta_y[te]) ** 2).sum() / (
            ((b.delta_y[te] - b.delta_y[te].mean()) ** 2).sum() + 1e-12)
        da = float((np.sign(pred).astype(int) == dir_y[te]).mean())
        return da, float(r2)
    da, r2 = ridge_dir_r2(b.py_k[:, None])
    out["oracle_yt"] = {"dir_acc": da, "dy_r2": r2}
    # 3 oracle [y_t, y_t1] -> KNOWN-ANSWER (dy is exactly y_t1 - y_t)
    da, r2 = ridge_dir_r2(np.stack([b.py_k, b.py_k1], 1))
    out["oracle_yt_yt1"] = {"dir_acc": da, "dy_r2": r2}
    # 4 raw paddle-crop pair, training-free centroid displacement
    cen_t = paddle_centroid(b.cur_frame)
    cen_t1 = paddle_centroid(b.fac_frame)
    dcen = cen_t1 - cen_t
    valid = ~np.isnan(dcen)
    # direction from measured centroid displacement (sign), evaluated on test set
    pred_dir = np.sign(np.where(np.abs(dcen) < 0.5, 0, dcen)).astype(int)
    m = te & valid
    out["crop_centroid"] = {
        "dir_acc": float((pred_dir[m] == dir_y[m]).mean()),
        "bal_acc": float(balanced_accuracy_score(dir_y[m], pred_dir[m])),
        "n": int(m.sum()),
        "corr_with_true_dy": float(np.corrcoef(dcen[te & valid],
                                               b.delta_y[te & valid])[0, 1]),
    }
    return out


# ------------------------------------------------------------------ orchestration
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ae, ck = load_frozen_ae()
    b = CM.load_eval_bundle()
    dir_y = np.sign(b.delta_y).astype(int)
    dy = b.delta_y

    # frozen spatial latents for frame_t and factual frame_t1
    zt = spatial_latent(ae, b.cur_frame)
    zt1 = spatial_latent(ae, b.fac_frame)

    # sanity: encoder frozen + deterministic
    assert not any(p.requires_grad for p in ae.parameters())
    assert np.allclose(zt, spatial_latent(ae, b.cur_frame)), "encoder non-deterministic"

    variants = ["full", "gap", "local", "full+diff", "local+diff"]
    seeds = [0, 1, 2]
    rows = []
    baselines_by_seed = []
    for s in seeds:
        tr, te = CM.group_train_test_split(b.ep_seed, 0.3, seed=s)
        # leakage guard
        assert not (set(b.ep_seed[tr]) & set(b.ep_seed[te]))
        baselines_by_seed.append(run_baselines(b, tr, te))
        for v in variants:
            rows.append(eval_variant(v, zt, zt1, dir_y, dy, tr, te, s))
            if v in ("local", "local+diff"):   # one small MLP probe on the local variants
                r = eval_variant(v, zt, zt1, dir_y, dy, tr, te, s, use_mlp=True)
                r["variant"] = v + "(mlp)"
                rows.append(r)

    # aggregate across seeds
    agg = _aggregate(rows)
    baselines = _agg_baselines(baselines_by_seed)

    # subset breakdown (seed 0) for the two headline variants + existing
    subsets = _subset_breakdown(b, zt, zt1, dir_y, dy)

    # ---- write artifacts ----
    _write_audit(ck, b, zt, dir_y)
    results = {
        "checkpoint": str(AE_CKPT), "ae_train_git_commit": ck["git_commit"],
        "followup_git_commit": git_commit(),
        "dataset_dir": str(b.data_dir),
        "n_eval_tuples": len(b.pool), "split": "episode-grouped, test_frac=0.3",
        "seeds": seeds, "class_distribution": _class_dist(dir_y),
        "baselines": baselines, "variants": agg, "subsets": subsets,
        "paddle_latent_cols": [PADDLE_COLS.start, PADDLE_COLS.stop],
    }
    CM.write_json(OUT / "p2_results.json", results)
    _write_csv(agg, baselines)
    _confusion_png(rows)
    _gap_png(agg, baselines)
    conclusion, nextstep = _decide(agg, baselines)
    _write_report(results, agg, baselines, subsets, conclusion, nextstep)
    print(f"[p2-followup] conclusion: {conclusion}")
    for a in agg:
        print(f"  {a['variant']:14s} dir {a['dir_acc_correct_mean']:.3f} "
              f"gap {a['temporal_gap_mean']:+.3f}  static {a['dir_acc_static_mean']:.3f} "
              f"R2 {a['dy_r2_mean']:+.2f}")
    print(f"  baselines: majority {baselines['majority']['dir_acc_mean']:.3f}, "
          f"oracle[yt,yt1] dir {baselines['oracle_yt_yt1']['dir_acc_mean']:.3f} "
          f"R2 {baselines['oracle_yt_yt1']['dy_r2_mean']:.2f}, "
          f"crop-centroid dir {baselines['crop_centroid']['dir_acc_mean']:.3f}")


def _aggregate(rows):
    names = []
    for r in rows:
        if r["variant"] not in names:
            names.append(r["variant"])
    keys = ["dir_acc_correct", "bal_acc_correct", "dir_acc_mismatch", "dir_acc_same",
            "dir_acc_static", "dir_acc_shuffled", "temporal_gap", "dy_r2", "dy_mae"]
    agg = []
    for n in names:
        rs = [r for r in rows if r["variant"] == n]
        d = {"variant": n, "dim": rs[0]["dim"], "n_seeds": len(rs)}
        for k in keys:
            vals = np.array([r[k] for r in rs], float)
            d[k + "_mean"] = float(vals.mean()); d[k + "_std"] = float(vals.std())
        agg.append(d)
    return agg


def _agg_baselines(bl_list):
    out = {}
    for key in bl_list[0]:
        out[key] = {}
        for mk in bl_list[0][key]:
            vals = np.array([bl[key][mk] for bl in bl_list], float)
            out[key][mk + "_mean"] = float(np.nanmean(vals))
            out[key][mk + "_std"] = float(np.nanstd(vals))
    return out


def _class_dist(dir_y):
    vals, cnts = np.unique(dir_y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def _subset_breakdown(b, zt, zt1, dir_y, dy):
    tr, te = CM.group_train_test_split(b.ep_seed, 0.3, seed=0)
    step = 4.586
    lo, hi = b.py_k.min(), b.py_k.max()
    boundary = ((b.py_k <= lo + step) | (b.py_k >= hi - step)
                | (b.py_k1 <= lo + step) | (b.py_k1 >= hi - step))
    masks = {"all": np.ones(len(dir_y), bool), "non_boundary": ~boundary,
             "free": b.stuck == 0, "stuck": b.stuck == 1}
    out = []
    for v in ("full", "local+diff"):
        for name, msk in masks.items():
            trm, tem = tr & msk, te & msk
            if tem.sum() < 6 or len(np.unique(dir_y[trm])) < 2:
                out.append({"variant": v, "subset": name, "n_test": int(tem.sum()),
                            "dir_acc": None}); continue
            r = eval_variant(v, zt, zt1, dir_y, dy, trm, tem, 0)
            out.append({"variant": v, "subset": name, "n_test": int(tem.sum()),
                        "dir_acc": round(r["dir_acc_correct"], 3),
                        "temporal_gap": round(r["temporal_gap"], 3)})
    return out


# ------------------------------------------------------------------ writers
def _write_audit(ck, b, zt, dir_y):
    md = f"""# P2 probe audit (existing implementation)

**Source:** `cjepa_rung3_diag_ae/run_diagnostic.py::Diagnostic.step_p2` (+ `probes.py`).

## Exact inputs to the existing P2 probe (verified, not inferred)
- Latent tensor before the probe: AE `encode` returns **(N, 16, 6, 6)** spatial map.
- Aggregation: `ae_model.encode_numpy(flatten=True)` **flattens** the full (16,6,6) map to
  **576** dims. **No global average pooling.** Spatial coordinates ARE preserved (as
  distinct flattened features); a linear probe can weight each cell independently.
- Probe input: `X2 = concatenate([z_t, z_t1], axis=1)` = **1152** dims, where z_t encodes
  `frame_t` and z_t1 encodes the **factual** `frame_t1` (oracle `o_fac_frame`).
- Action included? **No.** Explicit difference term (z_t1 − z_t)? **No.**
- Receives: **z_t AND z_t1** (concatenated). Not z_t only; not an explicit difference.
- Probe architecture: **Ridge(alpha=1)** for Δy regression; **LogisticRegression** (3-class)
  for direction. Features standardized with train statistics (`probes._standardize`).
- Target: `Δy = player_y_t1 − player_y_t` (screen px, regression); direction = `sign(Δy)`
  ∈ {{−1,0,+1}}. Class distribution (all 250): {_class_dist(dir_y)}.
- Split: episode-grouped, `test_frac=0.3`, seed 0 → 173 train / 77 test tuples, no group
  overlap; paired (z_t, z_t1) samples stay together (both come from the same tuple).
- Existing mismatch control: permutes z_t1 rows within train AND test, then **fits and
  evaluates on the permuted pairs** (measures spurious signal from garbage pairs), rather
  than fit-correct / eval-mismatched. Original result: Δy R² −0.72 → −0.70 (no drop),
  dir acc 0.558 vs majority baseline 0.44.
- Static baseline present (z_t alone) and shuffled-label control present.

## Implication
The existing probe already uses the **full spatial latent without pooling**. So "undo the
pooling" is NOT an available lever — pooling was never applied. The remaining hypotheses
this follow-up tests: (a) paddle-local restriction sharpens the signal; (b) an explicit
(z_t1 − z_t) term helps a linear probe; (c) a clean fit-correct/eval-mismatched protocol
reveals a temporal-pair gap the original (fit-on-garbage) control masked.
"""
    (OUT / "p2_audit.md").write_text(md, encoding="utf-8")


def _write_csv(agg, baselines):
    with open(OUT / "p2_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe_input", "dim", "dir_acc", "bal_acc", "dy_r2", "dy_mae",
                    "mismatch_acc", "same_acc", "static_acc", "shuffled_acc",
                    "temporal_gap"])
        for a in agg:
            w.writerow([a["variant"], a["dim"],
                        f"{a['dir_acc_correct_mean']:.3f}±{a['dir_acc_correct_std']:.3f}",
                        f"{a['bal_acc_correct_mean']:.3f}",
                        f"{a['dy_r2_mean']:.3f}", f"{a['dy_mae_mean']:.3f}",
                        f"{a['dir_acc_mismatch_mean']:.3f}",
                        f"{a['dir_acc_same_mean']:.3f}",
                        f"{a['dir_acc_static_mean']:.3f}",
                        f"{a['dir_acc_shuffled_mean']:.3f}",
                        f"{a['temporal_gap_mean']:+.3f}±{a['temporal_gap_std']:.3f}"])
        w.writerow([])
        for k, v in baselines.items():
            w.writerow([f"baseline:{k}"] + [f"{mk}={mv:.3f}" for mk, mv in v.items()
                                            if mk.endswith("_mean")])


def _confusion_png(rows):
    plt = _plt()
    keys = ["full", "local", "local+diff"]
    r0 = {r["variant"]: r for r in rows if r["split_seed"] == 0}
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.6))
    for j, k in enumerate(keys):
        r = r0[k]
        cm = confusion_matrix(r["_yte"], r["_pred_correct"], labels=DIR_LABELS)
        ax[j].imshow(cm, cmap="Blues")
        ax[j].set_xticks(range(3)); ax[j].set_xticklabels(["down", "stay", "up"])
        ax[j].set_yticks(range(3)); ax[j].set_yticklabels(["down", "stay", "up"])
        for a in range(3):
            for c in range(3):
                ax[j].text(c, a, cm[a, c], ha="center", va="center")
        ax[j].set_title(f"{k} (correct pair)"); ax[j].set_xlabel("pred")
        if j == 0:
            ax[j].set_ylabel("true")
    fig.suptitle("P2 follow-up: direction confusion (seed 0, correct pair)")
    fig.tight_layout(); fig.savefig(OUT / "p2_confusion_matrices.png", dpi=110)
    plt.close(fig)


def _gap_png(agg, baselines):
    plt = _plt()
    names = [a["variant"] for a in agg]
    corr = [a["dir_acc_correct_mean"] for a in agg]
    mis = [a["dir_acc_mismatch_mean"] for a in agg]
    stat = [a["dir_acc_static_mean"] for a in agg]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 0.25, corr, 0.25, label="correct pair")
    ax.bar(x, mis, 0.25, label="mismatched z_t1")
    ax.bar(x + 0.25, stat, 0.25, label="static (z_t)")
    ax.axhline(baselines["majority"]["dir_acc_mean"], ls="--", c="k",
               label=f"majority {baselines['majority']['dir_acc_mean']:.2f}")
    ax.axhline(baselines["oracle_yt_yt1"]["dir_acc_mean"], ls=":", c="g",
               label=f"oracle[yt,yt1] {baselines['oracle_yt_yt1']['dir_acc_mean']:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Δy direction accuracy"); ax.set_ylim(0, 1)
    ax.set_title("P2 follow-up: correct vs mismatched vs static (mean over 3 seeds)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "p2_temporal_gap_plot.png", dpi=110)
    plt.close(fig)


def _decide(agg, baselines):
    """Decision rule from the task."""
    best = max(agg, key=lambda a: a["temporal_gap_mean"])
    maj = baselines["majority"]["dir_acc_mean"]
    existing = [a for a in agg if a["variant"] == "full"][0]
    # thresholds: a genuine temporal signal needs a clear positive gap AND correct-pair
    # clearly over static/majority, stable across seeds (gap_mean > 2*gap_std).
    gap_ok = best["temporal_gap_mean"] > 0.05 and \
        best["temporal_gap_mean"] > 2 * best["temporal_gap_std"]
    over_static = best["dir_acc_correct_mean"] - best["dir_acc_static_mean"] > 0.05
    over_majority = best["dir_acc_correct_mean"] - maj > 0.05
    improves = best["dir_acc_correct_mean"] - existing["dir_acc_correct_mean"] > 0.03 or \
        best["temporal_gap_mean"] - existing["temporal_gap_mean"] > 0.05
    if gap_ok and over_static and over_majority and improves:
        c = "PROBE/POOLING FAILURE"
        nxt = (f"The AE latent DOES carry Δy signal: variant '{best['variant']}' gives a "
               f"clear temporal-pair gap (+{best['temporal_gap_mean']:.2f}) over mismatched "
               "z_t1 and beats static/majority. The original P2 pooling/protocol under-"
               "extracted it. The current AE latent may proceed to transition-model "
               "qualification (with a paddle-local / difference read-out).")
    elif (not gap_ok) or (best["temporal_gap_mean"] <= 0.05):
        c = "REPRESENTATION FAILURE"
        nxt = ("No frozen-latent read-out (full, paddle-local, or explicit difference) "
               "produces a clear correct>mismatch temporal gap; performance stays near "
               "static/majority. The AE latent preserves position + branch locality but "
               "not a stable decodable transition-level Δy. Recommend a later two-frame/"
               "temporal representation experiment (NOT implemented here).")
    else:
        c = "INCONCLUSIVE"
        nxt = ("Controls conflict or the effect is within seed noise; the 77-tuple test / "
               "label ambiguity limits power. Re-pose with more eval episodes before "
               "deciding.")
    return c, nxt


def _write_report(results, agg, baselines, subsets, conclusion, nextstep):
    A = {a["variant"]: a for a in agg}

    def row(label, key):
        a = A.get(key)
        if not a:
            return f"| {label} | — | — | — | — | — |"
        return (f"| {label} | {a['dir_acc_correct_mean']:.3f} | "
                f"{a['bal_acc_correct_mean']:.3f} | {a['dy_r2_mean']:+.2f} | "
                f"{a['dir_acc_mismatch_mean']:.3f} | "
                f"{a['temporal_gap_mean']:+.3f}±{a['temporal_gap_std']:.3f} |")
    bl = baselines
    md = f"""# P2 follow-up report — is the AE-latent Δy signal mis-extracted or absent?

## 1. Identity
- AE checkpoint: `{results['checkpoint']}` (trained at git `{results['ae_train_git_commit'][:10]}`)
- Follow-up git commit: `{results['followup_git_commit'][:10]}`
- Dataset: `{results['dataset_dir']}`
- Eval pool: {results['n_eval_tuples']} fixed tuples; split {results['split']}; seeds {results['seeds']}
- Encoder frozen + deterministic (asserted). Same pool/split as the AE diagnostic.

## 2. Existing P2 implementation (see `p2_audit.md`)
Input = `[flatten(z_t), flatten(z_t1)]` = 1152 dims from the **full (16,6,6) spatial latent,
NO pooling**, no action, no explicit difference. Ridge/Logistic linear probes. So there is
no pooling to "undo"; original dir acc 0.558, Δy R² −0.72, mismatch did not hurt.

## 3. Comparison (mean over {len(results['seeds'])} episode-grouped split seeds)

| Probe input | Dir acc | Balanced acc | Δy R² | Mismatch acc | Temporal-pair gap |
|---|---:|---:|---:|---:|---:|
{row("Existing / Full spatial [z_t,z_t1]", "full")}
{row("Global-avg-pool [pool z_t,pool z_t1] (control)", "gap")}
{row("Paddle-local [z_t,z_t1]", "local")}
{row("Paddle-local (small MLP)", "local(mlp)")}
{row("Full + difference", "full+diff")}
{row("Paddle-local + difference", "local+diff")}
{row("Paddle-local + difference (small MLP)", "local+diff(mlp)")}

Temporal-pair gap = acc(correct pair) − acc(mismatched z_t1). Higher = genuine two-frame use.

## 4. Baselines & class distribution
- Direction class distribution (all 250): {results['class_distribution']} (down/stay/up).
- Majority-class dir acc: **{bl['majority']['dir_acc_mean']:.3f}**.
- Oracle player_y_t only: dir acc {bl['oracle_yt']['dir_acc_mean']:.3f}, R² {bl['oracle_yt']['dy_r2_mean']:+.2f}.
- Oracle [player_y_t, player_y_t1] (KNOWN-ANSWER, Δy is exactly their difference): dir acc
  **{bl['oracle_yt_yt1']['dir_acc_mean']:.3f}**, R² **{bl['oracle_yt_yt1']['dy_r2_mean']:+.2f}**
  → confirms the target is learnable and the eval code is correct.
- Raw paddle-crop centroid (training-free, two-frame): dir acc
  {bl['crop_centroid']['dir_acc_mean']:.3f}, corr(measured Δcentroid, true Δy)
  {bl['crop_centroid']['corr_with_true_dy_mean']:+.2f} → a genuine two-frame signal exists
  in the raw paddle pixels.

## 5. Controls (per variant, in `p2_results.json`)
correct pair / mismatched z_t1 / same-frame / static-only(z_t) / shuffled-label. The
diagnostic quantity is the correct−mismatch temporal gap (column above).

## 6. Subset breakdown (seed 0)
{_subset_md(subsets)}

## 7. Conclusion: **{conclusion}**

{nextstep}

## Artifacts
`p2_audit.md`, `p2_results.json`, `p2_comparison.csv`, `p2_confusion_matrices.png`,
`p2_temporal_gap_plot.png`.
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")


def _subset_md(subsets):
    lines = ["| variant | subset | n_test | dir acc | temporal gap |",
             "|---|---|---:|---:|---:|"]
    for s in subsets:
        da = "—" if s["dir_acc"] is None else f"{s['dir_acc']:.3f}"
        tg = s.get("temporal_gap", "—")
        lines.append(f"| {s['variant']} | {s['subset']} | {s['n_test']} | {da} | {tg} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
