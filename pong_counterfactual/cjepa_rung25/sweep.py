"""Rung 2.5 driver: the coarsening sweep.

Re-runs the validated Rung 2 pipeline (ALE sticky actions at the FIXED level s, fs=2,
history(N) model, Gumbel-Max abduction, seed-replay oracle) at progressively coarser
model-facing state resolutions c in {1,2,4,8,16} px, and logs how the CF-over-IV
advantage collapses.

Locked design (skill):
  - s is FIXED at 0.5 -- read off the Rung 2 results: the gap was clearest there
    (gap 0.432 with 128 stuck steps, vs 0.196 @ s=0.25, 0.152 @ s=0.1).
  - Trajectories are collected ONCE at full fidelity (same seeds/params as Rung 2's
    s=0.5 run) and coarsened OFFLINE per level -> Invariant 2 (same eval tuples
    everywhere) holds by construction.
  - The oracle (seed replay) and all eval targets stay FULL-FIDELITY; model
    predictions are decoded to physical units (bin centers) before the L1 error.
  - One fresh HistModel per level, same architecture/hyperparams; delta-bins refit
    per level.
  - The eval rng schedule replicates Rung 2's exactly, so the c=1 anchor must
    reproduce the stored Rung 2 s=0.5 row to the digit (acceptance check A).

Mechanism probe (check D, the Problem-B evidence): per tuple we also cache the
full-fidelity "other-branch" next state -- what step k would have produced had the
sticky draw gone the OTHER way. On free draws that is replay with intended a_{k-1}
(the would-have-stuck outcome). A level-c COLLISION is when the factual outcome and
the other-branch outcome land in the SAME coarse player_y token: the observed token
then carries zero information about the noise, and abduction has nothing to grab.

Run:  .venv/Scripts/python.exe -m pong_counterfactual.cjepa_rung25.sweep [--no-history]
"""
import argparse
import json
import os

import numpy as np

from pong_counterfactual.cjepa_rung2.collect_rung2 import collect, UP, DOWN, FRAMESKIP
from pong_counterfactual.cjepa_rung2.oracle_rung2 import Oracle
from pong_counterfactual.cjepa_rung1.noise import state_vec
from pong_counterfactual.cjepa_rung1.history import valid_ks, HistModel
from pong_counterfactual.cjepa_rung1.gumbel_abduction import (
    gumbel_posterior, cf_token, intervention_token)
from pong_counterfactual.cjepa_rung25.coarsen import (
    coarsen, coarse_vecs, fit_level_discretizer, hist_features_c)

S = 0.5                         # FIXED sticky level (clearest Rung 2 gap; see header)
LEVELS = [1, 2, 4, 8, 16]       # coarsening bin widths (px); 1 = the Rung 2 anchor
N = 8                           # history window (Rung 2 value)
N_TRAIN = 40
N_EVAL = 14
T = 250
N_SAMPLES = 250

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, f"oracle_cache_s{S}.npz")


def opposite(a):
    return DOWN if a == UP else UP


def l1(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def build_eval_tuples(eval_eps):
    """The ONE fixed eval-tuple set, selected exactly as Rung 2 did (same rng)."""
    pool = [(i, k) for i, ep in enumerate(eval_eps)
            for k in valid_ks(ep, N, move_only=True)]
    rng0 = np.random.default_rng(7)
    if len(pool) > N_SAMPLES:
        pool = [pool[i] for i in rng0.choice(len(pool), N_SAMPLES, replace=False)]
    return pool


def oracle_pass(eval_eps, pool):
    """Cache every full-fidelity oracle quantity ONCE (level-independent):
    o_cf (CF ground truth), o_fac (no-op validity), stuck label, and the
    other-branch next state for the collision probe."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if z["pool"].tolist() == [list(t) for t in pool]:
            print(f"[oracle] reusing cache {CACHE}")
            return z["o_cf"], z["o_fac"], z["stuck"], z["other"]
        print("[oracle] cache stale (pool mismatch) -- recomputing")
    oracle = Oracle(S)
    o_cf = np.full((len(pool), 4), np.nan)
    o_fac = np.full((len(pool), 4), np.nan)
    other = np.full((len(pool), 4), np.nan)
    stuck = np.full(len(pool), -1, dtype=int)   # 1 stuck / 0 free / -1 unknown
    for i, (ei, k) in enumerate(pool):
        ep = eval_eps[ei]
        a_prime = opposite(ep.intended[k])
        cf = oracle.cf_next_state(ep.seed, ep.intended, k, a_prime)
        fac = oracle.factual_next_state(ep.seed, ep.intended, k)
        st = oracle.stuck_at(ep.seed, ep.intended, k)
        # other-branch: replay intended a_{k-1}; on a FREE draw this executes the
        # previous action = the would-have-stuck outcome. (On a stuck draw every
        # action executes a_{k-1} anyway, so the free branch is unreachable --
        # collisions are measured on free tuples only.)
        oth = oracle._replay_step(ep.seed, ep.intended, k, ep.intended[k - 1])
        if cf is not None:
            o_cf[i] = cf
        if fac is not None:
            o_fac[i] = fac
        if oth is not None:
            other[i] = oth
        if st is not None:
            stuck[i] = int(st)
        if (i + 1) % 50 == 0:
            print(f"[oracle] {i + 1}/{len(pool)} tuples replayed")
    oracle.close()
    np.savez(CACHE, pool=np.array([list(t) for t in pool]),
             o_cf=o_cf, o_fac=o_fac, stuck=stuck, other=other)
    print(f"[oracle] cached -> {CACHE}")
    return o_cf, o_fac, stuck, other


def run_level(c, train_eps, eval_eps, train_ks, pool, o_cf, o_fac, stuck, other,
              n_hist):
    """Train one fresh model in the level-c coarse world and run the FIXED tuples."""
    ctrain = [coarse_vecs(ep, c) for ep in train_eps]
    ceval = [coarse_vecs(ep, c) for ep in eval_eps]

    pairs = [(ctrain[i][k], ctrain[i][k + 1])
             for i, ep in enumerate(train_eps) for k in train_ks[i]]
    disc = fit_level_discretizer(pairs, c)

    X = np.asarray([hist_features_c(ctrain[i], ep.intended, k, n_hist)
                    for i, ep in enumerate(train_eps) for k in train_ks[i]])
    tok = np.asarray([disc.to_tokens(ctrain[i][k], ctrain[i][k + 1])
                      for i, ep in enumerate(train_eps) for k in train_ks[i]])
    model = HistModel(disc).fit(X, tok)

    # held-out token accuracy (check E: training sanity per level), as in Rung 2
    eval_ks = [valid_ks(ep, N) for ep in eval_eps]
    Xe = np.asarray([hist_features_c(ceval[i], ep.intended, k, n_hist)
                     for i, ep in enumerate(eval_eps) for k in eval_ks[i]])
    toke = np.asarray([disc.to_tokens(ceval[i][k], ceval[i][k + 1])
                       for i, ep in enumerate(eval_eps) for k in eval_ks[i]])
    sub = np.random.default_rng(0).choice(len(Xe), min(400, len(Xe)), replace=False)
    top1 = model.top1(Xe[sub], toke[sub])

    rng = np.random.default_rng(0)   # same schedule as Rung 2 -> exact c=1 anchor
    ecf_p, eiv_p, ecf4, eiv4 = [], [], [], []
    ecopy_p, cf_rep = [], []
    noop_repro, oracle_repro = [], []
    sf_cf, sf_iv, ns_cf, ns_iv = [], [], [], []
    coll_py, coll_all = [], []
    coll_flag = []                   # per tuple: 1/0 on free tuples, -1 where unknowable
    for i, (ei, k) in enumerate(pool):
        ep = eval_eps[ei]
        cv = ceval[ei]
        pos_k_c, pos_next_c = cv[k], cv[k + 1]
        a_int = ep.intended[k]
        a_prime = opposite(a_int)

        x_fac = hist_features_c(cv, ep.intended, k, n_hist)
        x_cf = hist_features_c(cv, ep.intended, k, n_hist, override_action=a_prime)
        lf, lc = model.logits(x_fac), model.logits(x_cf)
        obs = disc.to_tokens(pos_k_c, pos_next_c)

        cf_tok, iv_tok, noop_ok = [], [], True
        for d in range(4):
            g = gumbel_posterior(lf[d], obs[d], rng)              # ABDUCT at level c
            noop_ok &= (cf_token(lf[d], g) == obs[d])
            cf_tok.append(cf_token(lc[d], g))                     # CF: reuse g
            iv_tok.append(intervention_token(lc[d], rng))         # IV: fresh g
        noop_repro.append(noop_ok)

        m_cf = disc.decode(pos_k_c, cf_tok)     # decode to PHYSICAL units (bin centers)
        m_iv = disc.decode(pos_k_c, iv_tok)
        m_copy = disc.decode(pos_k_c, obs)      # DIAGNOSTIC baseline: copy the factual
        oc = o_cf[i]                            # FULL-FIDELITY oracle CF (never coarse)
        oracle_repro.append(bool(np.allclose(o_fac[i], state_vec(ep.states[k + 1]))))

        e_cf, e_iv = abs(m_cf[2] - oc[2]), abs(m_iv[2] - oc[2])
        ecf_p.append(e_cf); eiv_p.append(e_iv)
        ecopy_p.append(abs(m_copy[2] - oc[2]))
        cf_rep.append(cf_tok[2] == obs[2])      # CF merely repeats the observed token?
        ecf4.append(l1(m_cf, oc)); eiv4.append(l1(m_iv, oc))

        if stuck[i] >= 0:
            (sf_cf if stuck[i] else ns_cf).append(e_cf)
            (sf_iv if stuck[i] else ns_iv).append(e_iv)

        # mechanism probe: on FREE tuples, do factual and would-have-stuck outcomes
        # collide into the same coarse token?
        if stuck[i] == 0 and not np.isnan(other[i]).any():
            tok_oth = disc.to_tokens(pos_k_c, coarsen(other[i], c))
            coll_py.append(obs[2] == tok_oth[2])
            coll_all.append(obs == tok_oth)
            coll_flag.append(int(obs[2] == tok_oth[2]))
        else:
            coll_flag.append(-1)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "c": c, "s": S, "N": n_hist, "n": len(pool),
        "top1_player": float(top1[2]), "top1_all": [float(t) for t in top1],
        "nbins": [disc.nbins(d) for d in range(4)],
        "error_CF": mean(ecf_p), "error_IV": mean(eiv_p),
        "gap": mean(eiv_p) - mean(ecf_p),
        "error_CF4": mean(ecf4), "error_IV4": mean(eiv4),
        "gap4": mean(eiv4) - mean(ecf4),
        "noop_reproduces": mean(noop_repro), "oracle_reproduces": mean(oracle_repro),
        "n_stuck": int(np.sum(stuck == 1)), "n_free": int(np.sum(stuck == 0)),
        "stuck_CF": mean(sf_cf), "stuck_IV": mean(sf_iv),
        "stuck_gap": mean(sf_iv) - mean(sf_cf),
        "free_CF": mean(ns_cf), "free_IV": mean(ns_iv),
        "free_gap": mean(ns_iv) - mean(ns_cf),
        "collision_py": mean(coll_py), "n_collision": len(coll_py),
        "collision_all4": mean(coll_all),
        "error_COPY": mean(ecopy_p),            # copy-the-factual diagnostic baseline
        "cf_repeats_obs": mean(cf_rep),         # how often CF == observed token (py)
        # per-tuple records for the mechanism decomposition (same tuple order as pool)
        "tuples": {"e_cf": [float(x) for x in ecf_p],
                   "e_iv": [float(x) for x in eiv_p],
                   "e_copy": [float(x) for x in ecopy_p],
                   "collide": coll_flag},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-history", action="store_true",
                    help="secondary axis: history OFF (N=1 features; same eval tuples)")
    args = ap.parse_args()
    n_hist = 1 if args.no_history else N
    out = os.path.join(HERE, "results_rung25_nohist.json" if args.no_history
                       else "results_rung25.json")

    print(f"[collect] ONE full-fidelity collection at s={S}, fs={FRAMESKIP} "
          f"(Rung 2's exact seeds/params)")
    train_eps = collect(S, N_TRAIN, T=T, base_seed=0, policy_seed=1)
    eval_eps = collect(S, N_EVAL, T=T, base_seed=1000, policy_seed=2)
    train_ks = [valid_ks(ep, N) for ep in train_eps]   # N=8 windows ALWAYS (fixed pool)
    pool = build_eval_tuples(eval_eps)
    print(f"[collect] {sum(len(t) for t in train_ks)} train transitions, "
          f"{len(pool)} fixed eval tuples")

    o_cf, o_fac, stuck, other = oracle_pass(eval_eps, pool)
    bad = np.isnan(o_cf).any(axis=1) | np.isnan(o_fac).any(axis=1)
    assert not bad.any(), f"{bad.sum()} tuples lost oracle states; cannot keep pairing"

    # action-induced paddle delta at fs=2 (for stating the knee against it):
    # factual |player_y move| on free move-steps, full fidelity.
    deltas = [abs(state_vec(eval_eps[ei].states[k + 1])[2]
                  - state_vec(eval_eps[ei].states[k])[2])
              for i, (ei, k) in enumerate(pool) if stuck[i] == 0]
    paddle_step = float(np.mean(deltas))
    print(f"[probe] mean free-step |paddle delta| = {paddle_step:.2f} px "
          f"({int(np.sum(stuck == 1))} stuck / {int(np.sum(stuck == 0))} free)")

    rows = []
    for c in LEVELS:
        print(f"\n[level c={c}] refit bins, train fresh model, run fixed tuples ...")
        r = run_level(c, train_eps, eval_eps, train_ks, pool,
                      o_cf, o_fac, stuck, other, n_hist)
        rows.append(r)
        print(f"  nbins={r['nbins']}  top1_py={r['top1_player']*100:.1f}%  "
              f"CF={r['error_CF']:.3f} IV={r['error_IV']:.3f} gap={r['gap']:.3f}  "
              f"stuck_gap={r['stuck_gap']:.3f}  collision_py={r['collision_py']*100:.1f}%  "
              f"COPY={r['error_COPY']:.3f}  cf=obs {r['cf_repeats_obs']*100:.0f}%")

    with open(out, "w") as f:
        json.dump({"s": S, "frameskip": FRAMESKIP, "n_hist": n_hist,
                   "paddle_step_px": paddle_step,
                   "stuck": [int(x) for x in stuck], "rows": rows}, f, indent=2)
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    main()
