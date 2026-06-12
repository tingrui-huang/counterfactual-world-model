"""Stage 4 — engine + oracle on learned tokens: the Rung-3 headline.

gumbel_abduction.py UNCHANGED, applied per token slot (decision ⑤ made the M slots
independent categoricals exactly so this works):
  ABDUCT   : g_m = gumbel_posterior(logits_factual[m], observed_token[m])
  INTERVENE: logits under a' (history identical, current action swapped)
  PREDICT  : cf_m = argmax(logits_cf[m] + g_m);  IV baseline = fresh noise.

Scoring (decision ⑥, DUAL):
  token space (primary)  : the oracle's full-fidelity CF next FRAME is encoded
                           through the SAME frozen eye -> target tokens; error =
                           per-slot mismatch fraction (Hamming/M).
  physical px (secondary): predicted tokens -> resized-px player_y via the Stage-2
                           linear probe; compared to the oracle CF player_y. The
                           rung-1/2-comparable axis.

Acceptance protocol (Rung 2.5's, non-negotiable — the headline alone is gameable):
  A  s=0 collapse (err_CF ~= err_IV) + no-op identity (abduction reproduces the
     observed token per slot, both at s=0 and s>0)
  B  headline: err_CF < err_IV at s>0
  C  collision rate reported alongside; advantage must live on LOW-collision tuples
  D  copy baseline (predict next token = observed token) must NOT beat the CF
  E  stuck-split: advantage concentrates on oracle-labelled stuck steps
Oracle gates (replay determinism, fire-rate ~ s) were re-confirmed in stage0_oracle.

Outputs: results_rung3_<tok>.json + the one-plot headline figure (token + px panels,
collision and copy overlaid).

Run:  python -m pong_counterfactual.cjepa_rung3.stage4_eval --name M4_K64_res84
      [--smoke]
"""
import argparse
import json

import numpy as np
import torch

from pong_counterfactual.cjepa_rung1.gumbel_abduction import (
    gumbel_posterior, cf_token, intervention_token)
from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3.data import (load_split, build_eval_pool,
                                                  opposite)
from pong_counterfactual.cjepa_rung3.tokenizer import load_eye, encode_frames
from pong_counterfactual.cjepa_rung3.stage3_transition import (token_cache, feat,
                                                               load_transition)
from pong_counterfactual.cjepa_rung3.stage2_gate import onehot_codes


def probe_decode(codes, W, b):
    """(M,) codes -> resized-px 4-vector via the Stage-2 linear probe."""
    M = len(codes)
    x = onehot_codes(np.asarray(codes)[None, :], M, W.shape[0] // M)
    return x @ W + b


def evaluate_s(s, tok_name, trans_name, n_eval_eps, n_samples, device,
               eye, eye_cfg, W, b):
    d = C.data_dir(s)
    eval_eps, toks = token_cache(s, "eval", C.EVAL_BASE_SEED, n_eval_eps,
                                 tok_name, device)
    pool = build_eval_pool(eval_eps, C.N_HIST, n_samples)
    z = np.load(d / "oracle_cache.npz")
    assert np.array_equal(z["pool"], np.asarray(pool)), \
        f"oracle cache pool mismatch at s={s} — rerun stage0_oracle"
    stuck, o_cf_vec, other_vec = z["stuck"], z["o_cf"], z["other"]
    model = load_transition(s, tok_name, trans_name, device)
    M, K, N = eye_cfg.M, eye_cfg.K, C.N_HIST

    # token targets: oracle CF frames through the frozen eye (decision ⑥ primary);
    # collision flags: factual vs other-branch next frame, free tuples (check C)
    tgt_codes = encode_frames(eye, z["o_cf_frame"], device)
    oth_codes = encode_frames(eye, z["other_frame"], device)

    rng = np.random.default_rng(0)               # same abduction rng schedule as R2
    rows = []
    for i, (ei, k) in enumerate(pool):
        ep, tk = eval_eps[ei], toks[ei]
        a_prime = opposite(int(ep.intended[k]))
        x_fac = torch.from_numpy(feat(tk, ep.intended, k, N, M, K)).to(device)
        x_cf = torch.from_numpy(
            feat(tk, ep.intended, k, N, M, K, override_action=a_prime)).to(device)
        with torch.no_grad():
            lf = model(x_fac[None])[0].cpu().numpy().astype(np.float64)  # (M,K)
            lc = model(x_cf[None])[0].cpu().numpy().astype(np.float64)
        obs = tk[k + 1]                          # observed next tokens (factual)

        cf_t, iv_t, noop_ok = np.zeros(M, int), np.zeros(M, int), True
        for m in range(M):
            g = gumbel_posterior(lf[m], int(obs[m]), rng)        # ABDUCT per slot
            noop_ok &= (cf_token(lf[m], g) == int(obs[m]))       # no-op identity
            cf_t[m] = cf_token(lc[m], g)                         # CF: reuse g
            iv_t[m] = intervention_token(lc[m], rng)             # IV: fresh g

        tgt = tgt_codes[i]
        ham = lambda a: float((np.asarray(a) != tgt).mean())     # token-space error
        o_py = C.FRAME.to_resized(o_cf_vec[i])[2]                # full-fidelity CF py
        px = lambda a: float(abs(probe_decode(a, W, b)[0, 2] - o_py))
        collide = -1
        if stuck[i] == 0 and not np.isnan(other_vec[i]).any():
            collide = int(np.array_equal(obs, oth_codes[i]))
        rows.append({"stuck": int(stuck[i]), "collide": collide,
                     "noop_ok": bool(noop_ok),
                     "tok_cf": ham(cf_t), "tok_iv": ham(iv_t), "tok_copy": ham(obs),
                     "px_cf": px(cf_t), "px_iv": px(iv_t), "px_copy": px(obs)})

    mean = lambda key, sel=None: float(np.mean(
        [r[key] for r in rows if sel is None or sel(r)])) if rows else 0.0
    has = lambda sel: any(sel(r) for r in rows)
    res = {"s": s, "n": len(rows),
           "noop_identity": mean("noop_ok"),
           "oracle_replay_vec_ok": float(z["vec_ok"]),
           "oracle_replay_frame_ok": float(z["frame_ok"]),
           "stuck_rate": float(np.mean([r["stuck"] for r in rows
                                        if r["stuck"] >= 0] or [0.0]))}
    for sp in ("tok", "px"):
        res[f"{sp}_CF"] = mean(f"{sp}_cf")
        res[f"{sp}_IV"] = mean(f"{sp}_iv")
        res[f"{sp}_COPY"] = mean(f"{sp}_copy")
        res[f"{sp}_gap"] = res[f"{sp}_IV"] - res[f"{sp}_CF"]
        for lbl, sel in (("stuck", lambda r: r["stuck"] == 1),
                         ("free", lambda r: r["stuck"] == 0),
                         ("nocoll", lambda r: r["collide"] == 0),
                         ("coll", lambda r: r["collide"] == 1)):
            if has(sel):
                res[f"{sp}_{lbl}_CF"] = mean(f"{sp}_cf", sel)
                res[f"{sp}_{lbl}_IV"] = mean(f"{sp}_iv", sel)
                res[f"{sp}_{lbl}_gap"] = res[f"{sp}_{lbl}_IV"] - res[f"{sp}_{lbl}_CF"]
    free = [r for r in rows if r["collide"] >= 0]
    res["collision"] = float(np.mean([r["collide"] for r in free])) if free else 1.0
    res["n_stuck"] = sum(r["stuck"] == 1 for r in rows)
    res["n_free"] = sum(r["stuck"] == 0 for r in rows)
    return res


def figure(rows, tok_name, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ss = [r["s"] for r in rows]
    for ax, sp, unit in ((axes[0], "tok", "per-slot mismatch vs oracle-CF tokens"),
                         (axes[1], "px", "player_y L1 (resized px) vs oracle CF")):
        ax.plot(ss, [r[f"{sp}_CF"] for r in rows], "o-", label="err_CF (abduction)")
        ax.plot(ss, [r[f"{sp}_IV"] for r in rows], "s-", label="err_IV (fresh noise)")
        ax.plot(ss, [r[f"{sp}_COPY"] for r in rows], "^--",
                label="copy baseline (check D)")
        ax2 = ax.twinx()
        ax2.plot(ss, [r["collision"] for r in rows], "x:", color="gray",
                 label="collision rate (check C)")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("collision rate", color="gray")
        ax.set_xlabel("sticky probability s")
        ax.set_ylabel(unit)
        ax.set_title(f"{'TOKEN space (primary)' if sp == 'tok' else 'PHYSICAL px (secondary)'}")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Rung 3 headline — eye {tok_name} (err_CF vs err_IV, "
                 "+ copy & collision overlays)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[stage4] figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--trans", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_ev, n_samples = C.N_EVAL_EPS, C.N_SAMPLES
    trans_name = args.trans or C.TransitionConfig().name
    if args.smoke:
        o = C.smoke_overrides()
        n_ev, n_samples = o["n_eval_eps"], o["n_samples"]
        tok_name = args.name or o["tok"].name
        trans_name = args.trans or o["trans"].name
    else:
        tok_name = args.name or C.TokenizerConfig().name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eye, eye_cfg = load_eye(tok_name, device)
    ga = np.load(C.tok_dir(tok_name) / "gate_artifacts.npz")
    W, b = ga["probe_W"], ga["probe_b"]

    rows = [evaluate_s(s, tok_name, trans_name, n_ev, n_samples, device,
                       eye, eye_cfg, W, b) for s in C.SS_EVAL]

    r0 = next(r for r in rows if r["s"] == 0.0)
    pos = [r for r in rows if r["s"] > 0]
    checks = {
        "A_noop_identity": all(r["noop_identity"] > 0.999 for r in rows),
        "A_s0_collapse_tok": abs(r0["tok_gap"]) < 0.02,
        "B_headline_tok": all(r["tok_CF"] < r["tok_IV"] for r in pos),
        "B_headline_px": all(r["px_CF"] < r["px_IV"] for r in pos),
        "C_collision": {f"s={r['s']}": r["collision"] for r in rows},
        "C_gap_on_noncollided_tok": all(
            r.get("tok_nocoll_gap", 0) >= r.get("tok_coll_gap", 0) - 1e-9
            for r in pos if "tok_coll_gap" in r),
        "D_copy_does_not_win_tok": all(r["tok_CF"] <= r["tok_COPY"] + 1e-9
                                       for r in pos),
        "D_copy_does_not_win_px": all(r["px_CF"] <= r["px_COPY"] + 1e-9
                                      for r in pos),
        "E_stuck_concentration_tok": all(
            r.get("tok_stuck_gap", 0) > r.get("tok_free_gap", 0)
            for r in pos if r["n_stuck"] >= 5),
    }
    out = {"tokenizer": tok_name, "transition": trans_name,
           "paddle_step_resized_px": C.paddle_step_resized(),
           "rows": rows, "checks": checks}
    out_json = C.root() / f"results_rung3_{tok_name}.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[stage4] results -> {out_json}")

    print("\n[stage4] === Rung 3 headline ===")
    hdr = f"{'s':>5} | {'tok_CF':>7} {'tok_IV':>7} {'tok_COPY':>8} | " \
          f"{'px_CF':>6} {'px_IV':>6} {'px_COPY':>7} | {'coll':>5} {'noop':>5}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['s']:>5.2f} | {r['tok_CF']:>7.3f} {r['tok_IV']:>7.3f} "
              f"{r['tok_COPY']:>8.3f} | {r['px_CF']:>6.2f} {r['px_IV']:>6.2f} "
              f"{r['px_COPY']:>7.2f} | {r['collision']:>5.2f} "
              f"{r['noop_identity']*100:>4.0f}%")
    for r in pos:
        print(f"  s={r['s']}: stuck-split tok gap "
              f"{r.get('tok_stuck_gap', float('nan')):.3f} (stuck, n={r['n_stuck']}) "
              f"vs {r.get('tok_free_gap', float('nan')):.3f} (free); "
              f"collision-split gap {r.get('tok_nocoll_gap', float('nan')):.3f} "
              f"(no-coll) vs {r.get('tok_coll_gap', float('nan')):.3f} (collided)")
    print("\n[stage4] acceptance checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    figure(rows, tok_name, C.root() / f"rung3_headline_{tok_name}.png")


if __name__ == "__main__":
    main()
