"""Stage 0a — recollect Rung 2's episodes WITH pixels (CPU, run once, write to Drive).

Identical to Rung 2's collection (ALE sticky s, fs=2, same behavior policy, same
seeds) — the ONE single-stream policy rng is preserved, so trajectories are
byte-identical to Rung 2/2.5's at the same (base_seed, policy_seed). We additionally
grab the rendered frame at every step and store it preprocessed (decision ①).

The 4 ground-truth numbers are stored too, but they NEVER reach the model — they are
kept ONLY for the oracle pass (stage0_oracle) and the Stage-2 probes (hard pitfall).

Resume safety: one .npz per episode. Because the policy rng is a single sequential
stream (Rung-2 parity), a partial collection re-rolls the env from episode 0 — cheap
CPU — but only WRITES the missing shards; finished shards are never rewritten. If all
shards exist the stage is a no-op.

Run (Colab CPU runtime or locally):
  python -m pong_counterfactual.cjepa_rung3.stage0_collect [--smoke]
"""
import argparse

import numpy as np

from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung2.collect_rung2 import behavior_policy, NOOP
from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3.frames import preprocess, spec_dict
from pong_counterfactual.cjepa_rung3.data import (EpisodeData, ep_path, save_episode,
                                                  save_meta)


def vec_or_nan(st):
    """state dict -> 4-vector with NaN where invalid (storable, oracle/probe-only)."""
    ball, py, ey = st["ball"], st["player_y"], st["enemy_y"]
    if ball is None or py is None or ey is None:
        return np.full(4, np.nan)
    return np.array([ball[0], ball[1], py, ey], dtype=np.float64)


def collect_split(s, split, base_seed, policy_seed, n_eps, T):
    """Sequential collection with Rung-2's single policy-rng stream; writes only
    missing shards. Frames are captured via env.render() (rgb_array, headless)."""
    d = C.data_dir(s)
    missing = [i for i in range(n_eps)
               if not ep_path(d, split, base_seed + i).exists()]
    if not missing:
        print(f"[stage0] s={s} {split}: all {n_eps} shards present — skip")
        return
    print(f"[stage0] s={s} {split}: {len(missing)}/{n_eps} shards to write "
          f"(re-rolling the full stream for rng parity)")
    env = PongEnv(frameskip=C.FRAMESKIP, repeat_action_probability=s)
    prng = np.random.default_rng(policy_seed)
    for ep_i in range(n_eps):
        seed = base_seed + ep_i
        st = env.reset(seed=seed)
        vecs = [vec_or_nan(st)]
        frames = [preprocess(env.env.render())]
        intended = []
        last_int = NOOP
        for t in range(T):
            a_int = behavior_policy(st, prng, last_int)     # consumes the shared rng
            st, _, done = env.step(int(a_int))
            intended.append(int(a_int))
            vecs.append(vec_or_nan(st))
            frames.append(preprocess(env.env.render()))
            last_int = a_int
            if done:
                break
        if ep_i in missing:
            save_episode(d, split, EpisodeData(
                seed, np.asarray(intended, dtype=np.int64),
                np.asarray(vecs), np.stack(frames)))
            print(f"[stage0]   wrote {ep_path(d, split, seed).name} "
                  f"({len(intended)} steps)")
    env.close()
    save_meta(d, {"s": s, "frameskip": C.FRAMESKIP, "T": T,
                  "frame_spec": spec_dict(),
                  "splits": {"train": [C.TRAIN_BASE_SEED, n_eps],
                             "eval": [C.EVAL_BASE_SEED, n_eps]}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_tr, n_ev, T = C.N_TRAIN_EPS, C.N_EVAL_EPS, C.T
    if args.smoke:
        o = C.smoke_overrides()
        n_tr, n_ev, T = o["n_train_eps"], o["n_eval_eps"], o["T"]
    print(f"[stage0] root = {C.root()}")
    for s in C.SS_EVAL:
        collect_split(s, "train", C.TRAIN_BASE_SEED, C.TRAIN_POLICY_SEED, n_tr, T)
        collect_split(s, "eval", C.EVAL_BASE_SEED, C.EVAL_POLICY_SEED, n_ev, T)
    print("[stage0] done")


if __name__ == "__main__":
    main()
