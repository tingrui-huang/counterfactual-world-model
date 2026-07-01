"""Frozen-feature probes shared by P1 (position) / P2 (displacement).

Every probe:
  * fits ONLY on the episode-grouped train split, scores on the disjoint test split;
  * reports train + test (to expose overfit under p>>n high-dim latents);
  * runs a shuffled-label negative control (fit on permuted y) — a real signal must
    beat it;
  * bootstraps a test CI over per-sample scores.

Standardization: features are z-scored using TRAIN statistics only (no test leakage),
which matters because the AE latent and the JEPA one-hot live on very different scales.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor

from pong_counterfactual.cjepa_rung3_diag_ae.common import bootstrap_ci


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def reg_probe(X, y, tr, te, model_name="ridge", alpha=1.0, seed=0, standardize=True):
    """Regression probe. Returns dict with MAE, normalized MAE, R2, CIs, controls."""
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    if standardize:
        Xtr, Xte = _standardize(Xtr, Xte)
    if model_name == "ridge":
        m = Ridge(alpha=alpha)
    elif model_name == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(128,), max_iter=800,
                         random_state=seed, early_stopping=False)
    else:
        raise ValueError(model_name)
    m.fit(Xtr, ytr)
    pred_te = m.predict(Xte)
    pred_tr = m.predict(Xtr)
    ae_te = np.abs(pred_te - yte)
    lo, hi = bootstrap_ci(ae_te, seed=seed + 1)
    # baseline: predict train mean
    base_mae = float(np.abs(yte - ytr.mean()).mean())
    ss_res = float(((pred_te - yte) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    # shuffled-label control
    rng = np.random.default_rng(seed)
    ysh = ytr.copy(); rng.shuffle(ysh)
    msh = Ridge(alpha=alpha) if model_name == "ridge" else MLPRegressor(
        hidden_layer_sizes=(128,), max_iter=800, random_state=seed)
    msh.fit(Xtr, ysh)
    sh_mae = float(np.abs(msh.predict(Xte) - yte).mean())
    yrange = float(np.nanmax(y) - np.nanmin(y)) + 1e-12
    return {
        "model": model_name,
        "test_mae": float(ae_te.mean()),
        "test_mae_ci": [lo, hi],
        "test_nmae": float(ae_te.mean()) / yrange,
        "train_mae": float(np.abs(pred_tr - ytr).mean()),
        "test_r2": r2,
        "baseline_mae": base_mae,
        "shuffled_mae": sh_mae,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "pred_te": pred_te, "y_te": yte,
    }


def clf_probe(X, y, tr, te, seed=0, standardize=True, max_iter=2000):
    """Multiclass logistic probe. Returns acc, CI, majority baseline, shuffled control."""
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    if standardize:
        Xtr, Xte = _standardize(Xtr, Xte)
    m = LogisticRegression(max_iter=max_iter)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    correct = (pred == yte).astype(float)
    lo, hi = bootstrap_ci(correct, seed=seed + 2)
    # majority baseline
    vals, cnts = np.unique(ytr, return_counts=True)
    maj = vals[cnts.argmax()]
    base = float((yte == maj).mean())
    # shuffled control
    rng = np.random.default_rng(seed)
    ysh = ytr.copy(); rng.shuffle(ysh)
    msh = LogisticRegression(max_iter=max_iter); msh.fit(Xtr, ysh)
    sh = float((msh.predict(Xte) == yte).mean())
    return {
        "test_acc": float(correct.mean()),
        "test_acc_ci": [lo, hi],
        "train_acc": float((m.predict(Xtr) == ytr).mean()),
        "baseline_acc": base,
        "shuffled_acc": sh,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "pred_te": pred, "y_te": yte,
    }


def error_by_bin(pred, y, n_bins=8):
    """Mean |error| within equal-width bins of the TARGET (for position-error plots)."""
    lo, hi = np.min(y), np.max(y)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "mae": float(np.abs(pred[m] - y[m]).mean()) if m.any() else float("nan"),
        })
    return rows
