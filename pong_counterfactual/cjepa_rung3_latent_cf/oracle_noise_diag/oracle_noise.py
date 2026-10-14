"""Trusted oracle-noise builder (simulator-derived, NOT paddle-movement inferred).

The oracle noise is the two-probe sticky label `stuck ∈ {0=free, 1=stuck}` from
`cjepa_rung2.oracle_rung2.Oracle.stuck_at` (reset-replay; the exact detector the whole
Rung-2 line trusts). Eval-pool labels come from the cached `oracle_cache.npz`; train
labels are computed here (reset-replay) and cached, in the EXACT order of
`latent_adapter.build_train_transitions` so `z_t[i]` aligns with `stuck[i]`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3.data import load_split, valid_ks
from pong_counterfactual.cjepa_rung3 import config as RC
from pong_counterfactual.cjepa_rung2.oracle_rung2 import Oracle

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CACHE = OUT / "train_stuck_cache.npz"


def load_config():
    base = LA.load_config()
    base.update(yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8")))
    return base


def _iter_train_tuples(cfg):
    """(ep_seed, k, a_int, a_prev) in the EXACT order of build_train_transitions."""
    eps = load_split(LA._p(cfg["data_dir"]), "train", RC.TRAIN_BASE_SEED, RC.N_TRAIN_EPS)
    for ep in eps:
        for k in valid_ks(ep, N=2, move_only=True):
            yield ep, int(ep.seed), int(k), int(ep.intended[k]), int(ep.intended[k - 1])


def build_train_stuck(cfg=None, log=print):
    """Trusted stuck label per train transition (cached). Returns (stuck, key) where
    key = (seed,k) list used to assert alignment against build_train_transitions."""
    cfg = cfg or load_config()
    tuples = list(_iter_train_tuples(cfg))
    seeds = np.array([t[1] for t in tuples]); ks = np.array([t[2] for t in tuples])
    a_int = np.array([t[3] for t in tuples]); a_prev = np.array([t[4] for t in tuples])
    key_hash = hashlib.sha256(np.stack([seeds, ks]).tobytes()).hexdigest()[:16]

    if CACHE.exists():
        z = np.load(CACHE)
        if str(z["key_hash"]) == key_hash:
            log(f"[oracle-noise] train stuck cache hit ({len(z['stuck'])} tuples)")
            return z["stuck"], (seeds, ks, a_int, a_prev)
        log("[oracle-noise] cache stale — recomputing")

    log(f"[oracle-noise] computing trusted two-probe stuck for {len(tuples)} train "
        "transitions (reset-replay; one-time, cached) ...")
    oracle = Oracle(cfg["sticky_s"])
    stuck = np.empty(len(tuples), np.int64)
    for i, (ep, seed, k, ai, ap) in enumerate(tuples):
        s = oracle.stuck_at(seed, ep.intended, k)
        stuck[i] = -1 if s is None else int(s)
        if (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(tuples)}  (stuck rate so far "
                f"{(stuck[:i+1][stuck[:i+1]>=0]==1).mean():.3f})")
    oracle.close()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, stuck=stuck, seeds=seeds, ks=ks, a_int=a_int, a_prev=a_prev,
             key_hash=key_hash)
    log(f"[oracle-noise] cached -> {CACHE}  (stuck rate "
        f"{(stuck[stuck>=0]==1).mean():.3f})")
    return stuck, (seeds, ks, a_int, a_prev)


def assert_alignment(train, key):
    """train = build_train_transitions(); key = (seeds,ks,a_int,a_prev). The stuck labels
    are aligned to `train` iff seeds/actions match element-wise."""
    seeds, ks, a_int, a_prev = key
    assert np.array_equal(train.ep_seed, seeds), "seed order mismatch (stuck misaligned)"
    assert np.array_equal(train.a_int, a_int), "a_int mismatch"
    assert np.array_equal(train.a_prev, a_prev), "a_prev mismatch"


# ------------------------------------------------------------------ D0 (oracle-noise)
def gate_d0(cfg=None):
    cfg = cfg or load_config()
    checks = {}
    ev = LA.build_eval_tuples(cfg)
    # eval noise is the trusted cached two-probe label
    checks["eval_noise_from_cache"] = bool(set(np.unique(ev.stuck)) <= {-1, 0, 1})
    checks["eval_noise_binary_full"] = bool((ev.stuck >= 0).all())
    # CF meaning under swap: on stuck tuples the seed-replay CF latent == factual latent
    same = np.all(np.isclose(ev.z_t1, ev.z_cf_oracle), axis=1)
    checks["cf_equals_factual_on_stuck"] = bool(same[ev.stuck == 1].mean() > 0.95)
    checks["cf_differs_on_free"] = bool((~same[ev.stuck == 0]).mean() > 0.8)
    # train noise aligned + trusted
    train = LA.build_train_transitions(cfg)
    stuck_tr, key = build_train_stuck(cfg)
    assert_alignment(train, key)
    checks["train_noise_aligned"] = True
    checks["train_stuck_rate_near_s"] = bool(
        0.35 < (stuck_tr[stuck_tr >= 0] == 1).mean() < 0.65)   # ~ sticky s=0.5
    # wrong/shuffled-noise controls are genuinely different
    rng = np.random.default_rng(0)
    wrong = 1 - ev.stuck
    shuf = ev.stuck[rng.permutation(len(ev.stuck))]
    checks["wrong_noise_differs"] = bool((wrong != ev.stuck).mean() > 0.95)
    checks["shuffled_noise_differs"] = bool((shuf != ev.stuck).mean() > 0.3)
    # frozen encoder + determinism (via adapter D0)
    d0a = LA.gate_d0(cfg)
    checks["adapter_d0_pass"] = bool(d0a["pass"])
    passed = all(checks.values())
    return {"gate": "D0", "pass": bool(passed), "checks": checks,
            "n_train": int(len(train.z_t)),
            "train_stuck_rate": float((stuck_tr[stuck_tr >= 0] == 1).mean()),
            "eval_stuck_counts": {int(k): int(v) for k, v in
                                  zip(*np.unique(ev.stuck, return_counts=True))},
            "ae_train_git_commit": d0a["ae_train_git_commit"]}
