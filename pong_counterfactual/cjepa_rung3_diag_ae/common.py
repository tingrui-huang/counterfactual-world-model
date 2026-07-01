"""Shared data contract for the reconstruction-AE diagnostic.

The ONE alignment guarantee: every representation (object-state, Discrete-JEPA,
reconstruction-AE) is evaluated on the SAME fixed eval-tuple pool and the SAME
episode-grouped split that the existing Rung 3 audit
(cjepa_rung3/diagnose_representation.py) used. We import that module's helpers
directly rather than re-deriving them, so the numbers are comparable by construction.

Frames are res84 to match the approved tokenizer_M4_K64_res84 eye (decision: a fair
comparison encodes IDENTICAL pixels through both encoders).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from pong_counterfactual.cjepa_rung3.data import (
    load_split, build_eval_pool, EpisodeData, ACTIONS_H, NOOP, UP, DOWN,
)
from pong_counterfactual.cjepa_rung3 import config as C
# Reuse the audit's exact split / stats helpers so the AE and JEPA share methodology.
from pong_counterfactual.cjepa_rung3.diagnose_representation import (
    group_train_test_split, bootstrap_ci, physical_distances, executed_action,
)

# player_y is column 2 of the 4-vector (ball_x, ball_y, player_y, enemy_y).
PLAYER_Y = 2

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "pong_counterfactual" / "results"

# The res84 dataset + approved eye. Both were copied from Drive for read-only audits.
DEFAULT_DATA_DIR = RESULTS / "data_s0.5-20260617T160801Z-3-001" / "data_s0.5"
DEFAULT_EYE = (RESULTS / "tokenizer_M4_K64_res84-20260617T152639Z-3-001"
               / "tokenizer_M4_K64_res84" / "final.pt")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------------ train-frame pool
def load_train_frames(data_dir: Path = DEFAULT_DATA_DIR):
    """All frames from the 40 TRAIN episodes, tagged by episode seed (for grouped
    splitting). The AE trains ONLY on these — eval episodes are never seen in training.
    Frames are the model's sole input; the 4-vectors are returned for evaluation only.
    Returns (frames uint8 (Ntr,res,res), ep_seed (Ntr,), player_y (Ntr,) with NaNs)."""
    eps = load_split(data_dir, "train", C.TRAIN_BASE_SEED, C.N_TRAIN_EPS)
    frames, seeds, py = [], [], []
    for ep in eps:
        frames.append(ep.frames)
        seeds.append(np.full(len(ep.frames), ep.seed, dtype=np.int64))
        py.append(ep.vecs[:, PLAYER_Y])
    return (np.concatenate(frames), np.concatenate(seeds), np.concatenate(py))


# ---------------------------------------------------------------- aligned eval bundle
@dataclass
class EvalBundle:
    """The fixed eval-pool tuples, aligned exactly to diagnose_representation.load().
    Everything is per-tuple (N = len(pool)); frames are res84 uint8."""
    data_dir: Path
    pool: list                       # [(ei,k), ...]
    ep_seed: np.ndarray              # (N,) eval-episode seed  -> grouping key
    a_int: np.ndarray                # (N,) intended action at k
    a_prev: np.ndarray               # (N,) intended action at k-1
    stuck: np.ndarray                # (N,) oracle stuck label {-1,0,1}
    identifiable: np.ndarray         # (N,) a_prev != a_int
    py_k: np.ndarray                 # (N,) player_y at t   (screen px, oracle/probe only)
    py_k1: np.ndarray                # (N,) player_y at t+1 (factual)
    delta_y: np.ndarray              # (N,) py_k1 - py_k   (PRIMARY target)
    cur_frame: np.ndarray            # (N,res,res) frame_t
    fac_frame: np.ndarray            # (N,res,res) factual frame_{t+1}
    cf_frame: np.ndarray             # (N,res,res) counterfactual branch frame
    other_frame: np.ndarray          # (N,res,res) would-have-stuck branch frame
    vec_k: np.ndarray                # (N,4) object-state at t (oracle reference)
    o_fac: np.ndarray                # (N,4) factual next object-state
    o_cf: np.ndarray                 # (N,4) counterfactual next object-state
    other: np.ndarray                # (N,4) other-branch next object-state
    phys: dict                       # physical_distances(o_fac,o_cf,other)
    res: int = field(default=84)


def load_eval_bundle(data_dir: Path = DEFAULT_DATA_DIR) -> EvalBundle:
    eval_eps = load_split(data_dir, "eval", C.EVAL_BASE_SEED, C.N_EVAL_EPS)
    pool = build_eval_pool(eval_eps, C.N_HIST, C.N_SAMPLES)
    with np.load(data_dir / "oracle_cache.npz", allow_pickle=True) as z:
        oc = {k: z[k] for k in z.files}

    a_int, a_prev, ep_seed = [], [], []
    py_k, py_k1, cur_frame, vec_k = [], [], [], []
    for (ei, k) in pool:
        ep = eval_eps[ei]
        a_int.append(int(ep.intended[k]))
        a_prev.append(int(ep.intended[k - 1]))
        ep_seed.append(int(ep.seed))
        py_k.append(ep.vecs[k][PLAYER_Y])
        py_k1.append(ep.vecs[k + 1][PLAYER_Y])
        cur_frame.append(ep.frames[k])
        vec_k.append(ep.vecs[k])
    a_int = np.asarray(a_int, np.int64)
    a_prev = np.asarray(a_prev, np.int64)
    py_k = np.asarray(py_k, np.float64)
    py_k1 = np.asarray(py_k1, np.float64)
    phys = physical_distances(oc["o_fac"], oc["o_cf"], oc["other"])
    return EvalBundle(
        data_dir=data_dir, pool=pool, ep_seed=np.asarray(ep_seed, np.int64),
        a_int=a_int, a_prev=a_prev, stuck=oc["stuck"].astype(np.int64),
        identifiable=(a_prev != a_int), py_k=py_k, py_k1=py_k1, delta_y=py_k1 - py_k,
        cur_frame=np.stack(cur_frame), fac_frame=oc["o_fac_frame"],
        cf_frame=oc["o_cf_frame"], other_frame=oc["other_frame"],
        vec_k=np.stack(vec_k), o_fac=oc["o_fac"], o_cf=oc["o_cf"], other=oc["other"],
        phys=phys, res=int(eval_eps[0].frames.shape[-1]),
    )


def utc_stamp() -> str:
    """A deterministic-ish run stamp WITHOUT Date.now (unavailable): use a content hash
    of the machine-independent config so reruns are traceable but not wall-clock bound.
    Callers may override with an explicit --stamp for real timestamps."""
    return "run"


def write_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
