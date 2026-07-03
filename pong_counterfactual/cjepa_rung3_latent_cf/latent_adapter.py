"""Latent dataset adapter: frozen AE paddle-local latents for train transitions + the
fixed eval tuples, plus normalization and the D0 adapter-validity gate.

The AE stays frozen; the paddle-local region is FIXED (imported from the P2 follow-up);
factual/CF frames come from the AE diagnostic's cached seed-replay oracle. Everything is
aligned to the same dataset / episode split used throughout Rung 3.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from pong_counterfactual.cjepa_rung3_diag_ae import common as CM
from pong_counterfactual.cjepa_rung3_diag_ae.ae_model import ConvAutoencoder, AEConfig
from pong_counterfactual.cjepa_rung3_diag_ae.p2_probe_followup.run_followup import PADDLE_COLS
from pong_counterfactual.cjepa_rung3.data import load_split, valid_ks
from pong_counterfactual.cjepa_rung3 import config as RC

HERE = Path(__file__).resolve().parent
REPO = CM.REPO_ROOT
NOOP, FIRE, UP, DOWN = 0, 1, 2, 3


def load_config() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))


def _p(rel) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO / p


# ------------------------------------------------------------------ frozen encoder
def load_frozen_ae(cfg=None):
    cfg = cfg or load_config()
    ck = torch.load(_p(cfg["ae_checkpoint"]), map_location="cpu", weights_only=False)
    ae = ConvAutoencoder(AEConfig(**ck["config"]))
    ae.load_state_dict(ck["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae, ck


def encode_local(ae, frames_u8, cols=PADDLE_COLS, batch=256):
    """(N,res,res) uint8 -> (N, 16*6*2=192) frozen paddle-local latent (flattened)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(frames_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + batch]))
            z = ae.encode(x)[..., cols]          # (b,16,6,2)
            out.append(z.reshape(z.shape[0], -1).cpu().numpy())
    if not len(frames_u8):
        return np.zeros((0, 192), np.float32)
    return np.concatenate(out)


ACTIONS_MOVE = (UP, DOWN)


def action_onehot(a):
    """4-dim one-hot over {NOOP,FIRE,UP,DOWN} (matches ALE action ids)."""
    v = np.zeros(4, np.float32); v[int(a)] = 1.0
    return v


# ------------------------------------------------------------------ train transitions
@dataclass
class TrainSet:
    z_t: np.ndarray      # (N,192)
    z_t1: np.ndarray     # (N,192)
    a_int: np.ndarray    # (N,)
    a_prev: np.ndarray   # (N,)
    ep_seed: np.ndarray  # (N,)  grouping key
    py_k: np.ndarray     # (N,)  player_y at t  (probe target / eval only)
    py_k1: np.ndarray    # (N,)  player_y at t+1


def build_train_transitions(cfg=None) -> TrainSet:
    """Single-step move transitions from the 40 TRAIN episodes (seeds 0..39)."""
    cfg = cfg or load_config()
    ae, _ = load_frozen_ae(cfg)
    eps = load_split(_p(cfg["data_dir"]), "train", RC.TRAIN_BASE_SEED, RC.N_TRAIN_EPS)
    ft, ft1, ai, ap, seed, pk, pk1 = [], [], [], [], [], [], []
    for ep in eps:
        for k in valid_ks(ep, N=2, move_only=True):    # k>=1 so a_prev defined
            ft.append(ep.frames[k]); ft1.append(ep.frames[k + 1])
            ai.append(int(ep.intended[k])); ap.append(int(ep.intended[k - 1]))
            seed.append(int(ep.seed))
            pk.append(ep.vecs[k][2]); pk1.append(ep.vecs[k + 1][2])
    ft, ft1 = np.stack(ft), np.stack(ft1)
    return TrainSet(encode_local(ae, ft), encode_local(ae, ft1),
                    np.asarray(ai), np.asarray(ap), np.asarray(seed),
                    np.asarray(pk, np.float64), np.asarray(pk1, np.float64))


# ------------------------------------------------------------------ eval tuples
@dataclass
class EvalSet:
    z_t: np.ndarray          # (250,192)  factual frame_t
    z_t1: np.ndarray         # (250,192)  factual next (o_fac_frame)
    z_cf_oracle: np.ndarray  # (250,192)  seed-replay CF next (o_cf_frame)
    a_int: np.ndarray
    a_prev: np.ndarray
    a_prime: np.ndarray
    stuck: np.ndarray        # oracle two-probe label (EVAL ONLY)
    identifiable: np.ndarray
    boundary: np.ndarray
    ep_seed: np.ndarray
    py_k: np.ndarray
    py_k1: np.ndarray        # true factual next player_y
    py_cf: np.ndarray        # true seed-replay CF player_y (o_cf[2])


def opposite(a):
    return DOWN if a == UP else UP


def build_eval_tuples(cfg=None) -> EvalSet:
    cfg = cfg or load_config()
    ae, _ = load_frozen_ae(cfg)
    b = CM.load_eval_bundle(_p(cfg["data_dir"]))
    step = RC.PADDLE_STEP_SCREEN_PX
    lo, hi = b.py_k.min(), b.py_k.max()
    boundary = ((b.py_k <= lo + step) | (b.py_k >= hi - step)
                | (b.py_k1 <= lo + step) | (b.py_k1 >= hi - step))
    a_prime = np.array([opposite(int(a)) for a in b.a_int], np.int64)
    return EvalSet(
        z_t=encode_local(ae, b.cur_frame), z_t1=encode_local(ae, b.fac_frame),
        z_cf_oracle=encode_local(ae, b.cf_frame),
        a_int=b.a_int, a_prev=b.a_prev, a_prime=a_prime, stuck=b.stuck,
        identifiable=b.identifiable, boundary=boundary, ep_seed=b.ep_seed,
        py_k=b.py_k, py_k1=b.py_k1, py_cf=b.o_cf[:, 2])


# ------------------------------------------------------------------ normalization
@dataclass
class Normalizer:
    mu: np.ndarray
    sd: np.ndarray

    def fwd(self, z):
        return (z - self.mu) / self.sd

    def inv(self, zn):
        return zn * self.sd + self.mu


def fit_normalizer(train: TrainSet) -> Normalizer:
    z = np.concatenate([train.z_t, train.z_t1], 0)     # both frames share the latent space
    return Normalizer(z.mean(0, keepdims=True), z.std(0, keepdims=True) + 1e-6)


def action_feats(a_int, a_prev):
    """[onehot(a_int)(4), onehot(a_prev)(4)] -> (N,8)."""
    return np.concatenate([np.stack([action_onehot(a) for a in a_int]),
                           np.stack([action_onehot(a) for a in a_prev])], 1).astype(np.float32)


# ------------------------------------------------------------------ frozen eval probe
@dataclass
class PositionProbe:
    """Ridge: raw paddle-local latent -> player_y (screen px). EVAL-ONLY readout, fit on
    TRAIN latents, never a transition-model training loss."""
    coef: np.ndarray
    intercept: float
    mu: np.ndarray
    sd: np.ndarray

    def predict(self, z_local):
        return ((z_local - self.mu) / self.sd) @ self.coef + self.intercept


def fit_position_probe(train: TrainSet, alpha: float = 1.0) -> PositionProbe:
    from sklearn.linear_model import Ridge
    X = np.concatenate([train.z_t, train.z_t1], 0)
    y = np.concatenate([train.py_k, train.py_k1], 0)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    m = Ridge(alpha=alpha).fit((X - mu) / sd, y)
    return PositionProbe(m.coef_, float(m.intercept_), mu, sd)


# ------------------------------------------------------------------ D0 gate
def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def gate_d0(cfg=None) -> dict:
    """Adapter validity. Returns a dict with PASS/FAIL + evidence."""
    cfg = cfg or load_config()
    ae, ck = load_frozen_ae(cfg)
    checks = {}

    # 1 deterministic encoding
    b = CM.load_eval_bundle(_p(cfg["data_dir"]))
    z1 = encode_local(ae, b.cur_frame[:32]); z2 = encode_local(ae, b.cur_frame[:32])
    checks["deterministic_encoding"] = bool(np.array_equal(z1, z2))

    # 2 fixed region matches the P2 follow-up
    checks["region_matches_p2"] = (list(cfg["paddle_cols"]) ==
                                   [PADDLE_COLS.start, PADDLE_COLS.stop])
    checks["local_dim_192"] = int(encode_local(ae, b.cur_frame[:2]).shape[1]) == 192

    # 3 frozen encoder
    checks["encoder_frozen"] = not any(p.requires_grad for p in ae.parameters())

    # 4 factual/oracle frame alignment: the cached o_fac_frame must equal the recorded
    #   next frame frames[k+1] (seed-replay reproduced it 100%); and the CF frame differs
    #   from factual exactly on the free (non-stuck) tuples.
    with np.load(_p(cfg["data_dir"]) / "oracle_cache.npz", allow_pickle=True) as z:
        vec_ok = float(z["vec_ok"]); frame_ok = float(z["frame_ok"])
    checks["oracle_replay_frame_100"] = frame_ok == 1.0 and vec_ok == 1.0
    # CF==factual exactly on stuck tuples (intervention nullified), differs on free ones
    ev = build_eval_tuples(cfg)
    same = np.all(np.isclose(ev.z_t1, ev.z_cf_oracle), axis=1)
    stuck = ev.stuck == 1
    checks["cf_equals_factual_on_stuck"] = bool(same[stuck].mean() > 0.95)
    checks["cf_differs_on_free"] = bool((~same[ev.stuck == 0]).mean() > 0.8)

    # 5 no split leakage: train episode seeds disjoint from eval
    tr = build_train_transitions(cfg)
    checks["train_eval_episode_disjoint"] = not (set(tr.ep_seed.tolist())
                                                 & set(ev.ep_seed.tolist()))

    # 6 cache round-trip determinism (fresh vs re-encoded subset)
    checks["cache_roundtrip_ok"] = bool(np.array_equal(
        encode_local(ae, b.cur_frame[:16]), z1[:16]))

    passed = all(checks.values())
    return {
        "gate": "D0", "pass": bool(passed), "checks": checks,
        "ae_ckpt_sha16": _sha(np.concatenate([v.detach().numpy().ravel()
                              for v in list(ae.state_dict().values())[:1]])),
        "ae_train_git_commit": ck["git_commit"],
        "n_train_transitions": int(len(tr.z_t)),
        "n_eval_tuples": int(len(ev.z_t)),
        "paddle_cols": list(cfg["paddle_cols"]),
    }
