"""Stage 2 — validate the eye (cheap gate; THE SOUL of this rung).

Freeze the tokenizer; BEFORE any transition model is trained, run two minutes-fast
diagnostics:

1. COLLISION PROBE (Rung 2.5's orange curve, on the real learned eye). On FREE eval
   tuples (Rung 2.5 convention — at a stuck step the not-stuck branch is unreachable
   by seed replay), encode the factual next frame AND the would-have-stuck
   ("other-branch") next frame through the frozen eye. Collision = ALL M tokens
   identical: the observed token then carries zero bits about the sticky noise and
   abduction has nothing to grab.

2. LINEAR PROBE. Ridge regression from the M one-hot codes to the ground-truth
   positions in RESIZED px (fit on train split, scored on eval split). Says whether
   position info survived into the tokens at all; player_y MAE must beat the paddle's
   per-step move (~2.41 resized px), or stuck-vs-free is sub-resolution.

GATE RULE: collision <= COLLISION_MAX (0.25 = Rung 2.5's knee) AND probe player_y
MAE <= paddle_step. FAIL -> go back to Stage 1 and turn the Decision-④ knobs
(K up, M up, then Decision-① resolution). NO transition model until this passes.

Every config's row is appended to gate_log.json — that table IS the Problem-B
finding. Per-tuple collision flags + probe weights are saved for Stage 4.

Run:  python -m pong_counterfactual.cjepa_rung3.stage2_gate --name M4_K64_res84
      [--smoke]
"""
import argparse
import dataclasses
import json
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge

from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3.data import load_split, build_eval_pool
from pong_counterfactual.cjepa_rung3.tokenizer import load_eye, encode_frames

DIMS = ("ball_x", "ball_y", "player_y", "enemy_y")


def onehot_codes(codes, M, K):
    """(N,M) int codes -> (N, M*K) float one-hot features (slot-blocked)."""
    N = len(codes)
    X = np.zeros((N, M * K), dtype=np.float32)
    for m in range(M):
        X[np.arange(N), m * K + codes[:, m]] = 1.0
    return X


def probe_features(eps, model, device):
    """All valid frames of a split -> (one-hot features, resized-px targets)."""
    feats, tgts = [], []
    for ep in eps:
        valid = ~np.isnan(ep.vecs).any(axis=1)
        codes = encode_frames(model, ep.frames[valid], device)
        feats.append(onehot_codes(codes, model.cfg.M, model.cfg.K))
        tgts.append(np.stack([C.FRAME.to_resized(v) for v in ep.vecs[valid]]))
    return np.concatenate(feats), np.concatenate(tgts)


def run_gate(name, n_train_eps, n_eval_eps, n_samples, device):
    model, cfg = load_eye(name, device)
    d = C.data_dir(C.S_TRAIN)
    eval_eps = load_split(d, "eval", C.EVAL_BASE_SEED, n_eval_eps)
    pool = build_eval_pool(eval_eps, C.N_HIST, n_samples)
    z = np.load(d / "oracle_cache.npz")
    assert np.array_equal(z["pool"], np.asarray(pool)), \
        "oracle cache pool mismatch — rerun stage0_oracle"
    stuck, other_vec = z["stuck"], z["other"]

    # ---- 1. collision probe (free tuples only; see module docstring) -------------
    coll_flag = np.full(len(pool), -1, dtype=np.int64)   # 1/0 free, -1 unknowable
    fac_frames, oth_frames, free_ix = [], [], []
    for i, (ei, k) in enumerate(pool):
        if stuck[i] == 0 and not np.isnan(other_vec[i]).any():
            fac_frames.append(eval_eps[ei].frames[k + 1])
            oth_frames.append(z["other_frame"][i])
            free_ix.append(i)
    if free_ix:
        fac_codes = encode_frames(model, np.stack(fac_frames), device)
        oth_codes = encode_frames(model, np.stack(oth_frames), device)
        same = (fac_codes == oth_codes)
        coll_flag[np.asarray(free_ix)] = same.all(axis=1).astype(np.int64)
        collision = float(same.all(axis=1).mean())
        collision_slot = [float(x) for x in same.mean(axis=0)]
    else:
        collision, collision_slot = 1.0, [1.0] * cfg.M
    n_free = len(free_ix)

    # ---- 2. linear probe (fit train split, score eval split) ---------------------
    train_eps = load_split(d, "train", C.TRAIN_BASE_SEED, n_train_eps)
    Xtr, Ytr = probe_features(train_eps, model, device)
    Xev, Yev = probe_features(eval_eps, model, device)
    ridge = Ridge(alpha=1.0).fit(Xtr, Ytr)
    mae = np.abs(ridge.predict(Xev) - Yev).mean(axis=0)
    probe_mae = {dim: float(m) for dim, m in zip(DIMS, mae)}

    # ---- codebook usage (token-budget sanity) -------------------------------------
    all_codes = np.concatenate(
        [encode_frames(model, ep.frames, device) for ep in train_eps])
    used = len(np.unique(all_codes))
    per_slot = [len(np.unique(all_codes[:, m])) for m in range(cfg.M)]

    # ---- the gate ------------------------------------------------------------------
    step_px = C.paddle_step_resized()
    pass_coll = collision <= C.COLLISION_MAX
    pass_probe = probe_mae["player_y"] <= step_px * C.PROBE_PY_MAX_FACTOR
    passed = bool(pass_coll and pass_probe)

    report = {
        "name": name, "config": dataclasses.asdict(cfg),
        "collision_allM": collision, "collision_per_slot": collision_slot,
        "n_free_tuples": n_free, "n_stuck": int((stuck == 1).sum()),
        "probe_mae_resized_px": probe_mae, "paddle_step_resized_px": step_px,
        "codes_used_of_K": used, "codes_used_per_slot": per_slot,
        "gate": {"collision_max": C.COLLISION_MAX, "pass_collision": pass_coll,
                 "probe_py_max": step_px * C.PROBE_PY_MAX_FACTOR,
                 "pass_probe": bool(pass_probe), "PASSED": passed},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    gd = C.tok_dir(name)
    (gd / "gate_report.json").write_text(json.dumps(report, indent=2))
    np.savez(gd / "gate_artifacts.npz",
             pool=np.asarray(pool), coll_flag=coll_flag,
             probe_W=ridge.coef_.T, probe_b=ridge.intercept_)
    log = C.root() / "gate_log.json"
    rows = json.loads(log.read_text()) if log.exists() else []
    rows.append({k: report[k] for k in
                 ("name", "collision_allM", "probe_mae_resized_px",
                  "codes_used_of_K", "time")} | {"PASSED": passed})
    log.write_text(json.dumps(rows, indent=2))

    print(f"\n[stage2] === GATE for {name} ===")
    print(f"  collision (all {cfg.M} tokens identical, {n_free} free tuples): "
          f"{collision*100:.1f}%   (max {C.COLLISION_MAX*100:.0f}%)  "
          f"per-slot same: {[f'{x:.2f}' for x in collision_slot]}")
    print(f"  linear-probe MAE (resized px): " +
          "  ".join(f"{k}={v:.2f}" for k, v in probe_mae.items()))
    print(f"  paddle step = {step_px:.2f} resized px; codebook used {used}/{cfg.K} "
          f"(per slot {per_slot})")
    print(f"  --> GATE {'PASSED' if passed else 'FAILED'}"
          + ("" if passed else "  -> turn knobs: K up / M up / loss weights / res ①;"
                              " do NOT train Stage 3"))
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_tr, n_ev, n_samples = C.N_TRAIN_EPS, C.N_EVAL_EPS, C.N_SAMPLES
    if args.smoke:
        o = C.smoke_overrides()
        n_tr, n_ev, n_samples = o["n_train_eps"], o["n_eval_eps"], o["n_samples"]
        name = args.name or o["tok"].name
    else:
        name = args.name or C.TokenizerConfig().name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_gate(name, n_tr, n_ev, n_samples, device)


if __name__ == "__main__":
    main()
