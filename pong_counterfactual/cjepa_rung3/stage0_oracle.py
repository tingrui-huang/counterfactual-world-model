"""Stage 0b — the PIXEL oracle pass (CPU, run once per s, cached to Drive).

Seed-replay oracle exactly as Rung 2 (same s, same fs, same prefix replay), but each
replayed next-state is captured BOTH as the full-fidelity 4-vector AND as the
preprocessed next FRAME, because Rung 3's primary metric lives in token space: the
oracle CF frame is later encoded through the frozen eye (decision ⑥).

Per fixed eval tuple (ei, k) we cache:
  o_cf   vec+frame : replay to k, step a' = opposite(intended[k])  (the CF target)
  o_fac  vec+frame : replay to k, step intended[k]   (replay-determinism gate)
  other  vec+frame : replay to k, step intended[k-1] (the would-have-stuck branch,
                     for the Stage-2 collision probe — measured on FREE tuples,
                     Rung 2.5 convention: at a STUCK step every action executes
                     a_{k-1}, so the not-stuck branch is unreachable by replay;
                     free steps give the same factual-vs-other-branch contrast)
  stuck  label     : two-probe detector (oracle-side only, NEVER a model input)

The oracle replays at FULL pixel fidelity; preprocessing of the captured frame is the
same decision-① transform the model's own observations get (the eye needs its input
format) — the 4-vectors and the replay itself are never coarsened (hard pitfall).

Resume safety: partial cache written every 25 tuples; on restart we continue from
the last partial. Final artifact: data_s{s}/oracle_cache.npz

Run:  python -m pong_counterfactual.cjepa_rung3.stage0_oracle [--smoke]
"""
import argparse
from pathlib import Path

import numpy as np

from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3.frames import preprocess
from pong_counterfactual.cjepa_rung3.data import (load_split, build_eval_pool,
                                                  opposite, NOOP, UP, DOWN)

MOVES = [NOOP, UP, DOWN]
SAVE_EVERY = 25


class PixelOracle:
    """Rung-2 seed-replay oracle, returning (4-vector, preprocessed frame)."""

    def __init__(self, s):
        self.env = PongEnv(frameskip=C.FRAMESKIP, repeat_action_probability=s)

    def replay_step(self, seed, intended, k, a_k):
        st = self.env.reset(seed=int(seed))
        for t in range(k):
            st, _, _ = self.env.step(int(intended[t]))
        st2, _, _ = self.env.step(int(a_k))
        from pong_counterfactual.cjepa_rung3.stage0_collect import vec_or_nan
        return vec_or_nan(st2), preprocess(self.env.env.render())

    def stuck_at(self, seed, intended, k):
        """Two-probe stuck detector (Rung 2): both probes != a_{k-1}; identical
        outcomes <=> the sticky fired. -1 if ill-posed (invalid probe states)."""
        a_prev = int(intended[k - 1])
        p1, p2 = [a for a in MOVES if a != a_prev][:2]
        n1, _ = self.replay_step(seed, intended, k, p1)
        n2, _ = self.replay_step(seed, intended, k, p2)
        if np.isnan(n1).any() or np.isnan(n2).any():
            return -1
        return int(bool(np.allclose(n1, n2)))

    def close(self):
        self.env.close()


def oracle_pass(s, n_eval_eps, n_samples, N):
    d = C.data_dir(s)
    final = d / "oracle_cache.npz"
    partial = d / "oracle_partial.npz"
    eval_eps = load_split(d, "eval", C.EVAL_BASE_SEED, n_eval_eps)
    pool = build_eval_pool(eval_eps, N, n_samples)
    pool_arr = np.asarray(pool, dtype=np.int64)
    res = eval_eps[0].frames.shape[-1]
    n = len(pool)

    if final.exists():
        z = np.load(final)
        if np.array_equal(z["pool"], pool_arr):
            print(f"[oracle] s={s}: cache complete ({n} tuples) — skip")
            return
        print(f"[oracle] s={s}: cache stale (pool mismatch) — recomputing")

    # fresh arrays, or resume from the partial
    arrs = {k: np.full((n, 4), np.nan) for k in ("o_cf", "o_fac", "other")}
    for k in ("o_cf_frame", "o_fac_frame", "other_frame"):
        arrs[k] = np.zeros((n, res, res), dtype=np.uint8)
    arrs["stuck"] = np.full(n, -1, dtype=np.int64)
    start = 0
    if partial.exists():
        z = np.load(partial)
        if np.array_equal(z["pool"], pool_arr):
            for k in arrs:
                arrs[k] = z[k].copy()
            start = int(z["n_done"])
            print(f"[oracle] s={s}: resuming at tuple {start}/{n}")

    oracle = PixelOracle(s)
    for i in range(start, n):
        ei, k = pool[i]
        ep = eval_eps[ei]
        a_prime = opposite(int(ep.intended[k]))
        for key, a in (("o_cf", a_prime), ("o_fac", int(ep.intended[k])),
                       ("other", int(ep.intended[k - 1]))):
            v, f = oracle.replay_step(ep.seed, ep.intended, k, a)
            arrs[key][i] = v
            arrs[key + "_frame"][i] = f
        arrs["stuck"][i] = oracle.stuck_at(ep.seed, ep.intended, k)
        if (i + 1) % SAVE_EVERY == 0 or i == n - 1:
            tmp = partial.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, pool=pool_arr, n_done=i + 1, **arrs)
            tmp.replace(partial)
            print(f"[oracle] s={s}: {i + 1}/{n} tuples replayed (partial saved)")
    oracle.close()

    # replay-determinism gate (inherited from Rung 2, re-confirmed once): the factual
    # replay must reproduce the recorded next-state AND the recorded next-frame.
    vec_ok, frame_ok = [], []
    for i, (ei, k) in enumerate(pool):
        ep = eval_eps[ei]
        vec_ok.append(bool(np.allclose(arrs["o_fac"][i], ep.vecs[k + 1],
                                       equal_nan=True)))
        frame_ok.append(bool(np.array_equal(arrs["o_fac_frame"][i],
                                            ep.frames[k + 1])))
    print(f"[oracle] s={s}: replay reproduces factual vec {np.mean(vec_ok)*100:.1f}% "
          f"/ frame {np.mean(frame_ok)*100:.1f}% (must be 100/100)")
    fire = float(np.mean(arrs["stuck"][arrs["stuck"] >= 0])) if (arrs["stuck"] >= 0).any() else 0.0
    print(f"[oracle] s={s}: stuck rate on move-tuples = {fire:.3f} (expect ~ s)")

    tmp = final.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pool=pool_arr, vec_ok=np.mean(vec_ok),
                        frame_ok=np.mean(frame_ok), **arrs)
    tmp.replace(final)
    partial.unlink(missing_ok=True)
    print(f"[oracle] s={s}: cached -> {final}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_ev, n_samples = C.N_EVAL_EPS, C.N_SAMPLES
    if args.smoke:
        o = C.smoke_overrides()
        n_ev, n_samples = o["n_eval_eps"], o["n_samples"]
    for s in C.SS_EVAL:
        oracle_pass(s, n_ev, n_samples, C.N_HIST)
    print("[oracle] done")


if __name__ == "__main__":
    main()
