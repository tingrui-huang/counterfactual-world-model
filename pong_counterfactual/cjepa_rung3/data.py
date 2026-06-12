"""Episode shards on Drive + the fixed eval-tuple pool.

One .npz per episode (resume-safe collection: finished episodes are never re-rolled):
  intended : (T,)   int      per-step intended actions (oracle replays these)
  vecs     : (T+1,4) float64 ground-truth 4-vectors, NaN rows where invalid.
             ORACLE/PROBE-ONLY — never a model input (hard pitfall).
  frames   : (T+1,res,res) uint8 preprocessed pixels — the model's ONLY observation.
  seed     : ()     int

valid_ks / pool selection replicate Rung 2/2.5's logic and rng EXACTLY (same seeds ->
same trajectories -> same tuples), so Rung 3 numbers are comparable across rungs.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

NOOP, FIRE, UP, DOWN = 0, 1, 2, 3
ACTIONS_H = (NOOP, UP, DOWN)


@dataclass
class EpisodeData:
    seed: int
    intended: np.ndarray     # (T,) int
    vecs: np.ndarray         # (T+1,4) float64, NaN where invalid (oracle/probe only)
    frames: np.ndarray       # (T+1,res,res) uint8


def ep_path(d: Path, split: str, seed: int) -> Path:
    return d / f"ep_{split}_{seed:05d}.npz"


def save_episode(d: Path, split: str, ep: EpisodeData):
    p = ep_path(d, split, ep.seed)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, seed=ep.seed, intended=ep.intended,
                        vecs=ep.vecs, frames=ep.frames)
    tmp.replace(p)          # atomic-ish: never leave a half-written shard as final


def load_episode(d: Path, split: str, seed: int) -> EpisodeData:
    z = np.load(ep_path(d, split, seed))
    return EpisodeData(int(z["seed"]), z["intended"], z["vecs"], z["frames"])


def load_split(d: Path, split: str, base_seed: int, n_eps: int) -> List[EpisodeData]:
    return [load_episode(d, split, base_seed + i) for i in range(n_eps)]


def save_meta(d: Path, meta: dict):
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def valid_ks(ep: EpisodeData, N: int, move_only: bool = False) -> List[int]:
    """Steps k usable as a transition with an N-window of valid observations.
    Mirrors cjepa_rung1.history.valid_ks (window k-N+1 .. k+1 all valid)."""
    out = []
    for k in range(N - 1, len(ep.intended)):
        a = int(ep.intended[k])
        if a not in ACTIONS_H:
            continue
        if move_only and a not in (UP, DOWN):
            continue
        if np.isnan(ep.vecs[k - N + 1:k + 2]).any():
            continue
        out.append(k)
    return out


def build_eval_pool(eval_eps: List[EpisodeData], N: int,
                    n_samples: int) -> List[Tuple[int, int]]:
    """The ONE fixed eval-tuple set — same selection rng as Rung 2/2.5 (rng(7))."""
    pool = [(i, k) for i, ep in enumerate(eval_eps)
            for k in valid_ks(ep, N, move_only=True)]
    rng0 = np.random.default_rng(7)
    if len(pool) > n_samples:
        pool = [pool[i] for i in rng0.choice(len(pool), n_samples, replace=False)]
    return pool


def opposite(a: int) -> int:
    return DOWN if a == UP else UP
