"""Rung 2.5 analysis: the collapse table, the knee, the mechanism check, the plot,
and the Rung 3 spec paragraph.

Run after sweep.py:  .venv/Scripts/python.exe -m pong_counterfactual.cjepa_rung25.analyze
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_rung25.json")
RESULTS_NOHIST = os.path.join(HERE, "results_rung25_nohist.json")
PLOT = os.path.join(HERE, "rung25_collapse.png")

# Rung 2 stored headline @ s=0.5 (the anchor must reproduce this within seed noise)
RUNG2_S05 = {"error_CF": 1.980, "error_IV": 2.412, "gap": 0.432, "stuck_gap": 0.64}
NOISE_FLOOR = 0.05   # |gap| at Rung 2's s=0 collapse control was 0.048


def load(path):
    with open(path) as f:
        return json.load(f)


def table(rows, gap1):
    print(f"    {'c':>3} | {'error_CF':>8} | {'error_IV':>8} | {'gap':>6} | "
          f"{'gap/gap(1)':>10} | {'stuck gap':>9} | {'collide_py':>10} | "
          f"{'top1_py':>7} | nbins")
    print("    " + "-" * 92)
    for r in rows:
        print(f"    {r['c']:>3} | {r['error_CF']:>8.3f} | {r['error_IV']:>8.3f} | "
              f"{r['gap']:>6.3f} | {r['gap'] / gap1:>10.2f} | {r['stuck_gap']:>9.3f} | "
              f"{r['collision_py'] * 100:>9.1f}% | {r['top1_player'] * 100:>6.1f}% | "
              f"{r['nbins']}")


def knee_of(rows, gap1):
    for r in rows:
        if r["gap"] / gap1 < 0.5:
            return r["c"]
    return None


def main():
    data = load(RESULTS)
    rows = data["rows"]
    paddle_step = data["paddle_step_px"]
    gap1 = rows[0]["gap"]

    print("=" * 78)
    print("CausalJEPA Rung 2.5 — coarsening sweep (Problem B probe)")
    print(f"s={data['s']} fixed, fs={data['frameskip']}, N={data['n_hist']}, "
          f"n={rows[0]['n']} fixed eval tuples at every level")
    print("=" * 78)

    # ---- A: anchor ---------------------------------------------------------------
    r1 = rows[0]
    ok_anchor = (abs(r1["error_CF"] - RUNG2_S05["error_CF"]) < 0.15
                 and abs(r1["error_IV"] - RUNG2_S05["error_IV"]) < 0.15
                 and r1["gap"] > 0)
    print(f"\n[A] Anchor: c=1 vs stored Rung 2 s=0.5 row "
          f"(CF {RUNG2_S05['error_CF']}, IV {RUNG2_S05['error_IV']}, gap {RUNG2_S05['gap']}):")
    print(f"    c=1 -> CF {r1['error_CF']:.3f}, IV {r1['error_IV']:.3f}, "
          f"gap {r1['gap']:.3f}, stuck_gap {r1['stuck_gap']:.3f}   "
          f"{'REPRODUCED' if ok_anchor else '** MISMATCH — harness broken, stop **'}")

    # ---- consistency at every level (model no-op + oracle no-op) ------------------
    ok_noop = all(r["noop_reproduces"] > 0.999 and r["oracle_reproduces"] > 0.999
                  for r in rows)
    print(f"\n[inv] model no-op CF == observed token AND oracle no-op == factual, all "
          f"levels: {'100% everywhere' if ok_noop else '** VIOLATED **'}")

    # ---- C: the collapse curve -----------------------------------------------------
    print(f"\n[C] Collapse table (errors in px vs FULL-FIDELITY oracle CF; "
          f"gap = IV - CF):")
    table(rows, gap1)
    knee = knee_of(rows, gap1)
    floor = next((r["c"] for r in rows if abs(r["gap"]) < NOISE_FLOOR), None)
    mono = all(rows[i]["gap"] >= rows[i + 1]["gap"] - 0.05 for i in range(len(rows) - 1))
    print(f"\n    knee c* (first gap/gap(1) < 0.5): {knee} px   "
          f"(free-step paddle delta at fs=2: {paddle_step:.2f} px)")
    print(f"    noise floor (|gap| < {NOISE_FLOOR}): "
          f"{'c=' + str(floor) if floor else 'not reached'}")
    print(f"    monotone-ish decay: {mono}")

    # ---- D: mechanism check --------------------------------------------------------
    # the NOISE-RECOVERY gap = IV-CF on non-collided free tuples (the only tuples whose
    # observed token still carries noise information). The RAW gap is contaminated at
    # coarse c by the copying artifact (D2/D3), so the mechanism correlate is gap_nc.
    coll = np.array([r["collision_py"] for r in rows])
    gap_nc = []
    for r in rows:
        t = r["tuples"]
        e_cf = np.array(t["e_cf"]); e_iv = np.array(t["e_iv"])
        nc = np.array(t["collide"]) == 0
        gap_nc.append(float(e_iv[nc].mean() - e_cf[nc].mean()) if nc.any() else np.nan)
    gap_nc = np.array(gap_nc)

    print(f"\n[D] Mechanism (collapse = information destruction): stuck/free outcome-")
    print(f"    token collision rate on free tuples (n={rows[0]['n_collision']}) "
          f"must TRACK the destruction of the noise-recovery gap:")
    for r, gnc in zip(rows, gap_nc):
        bar_g = "#" * int(max(r["gap"] / gap1, 0) * 30)
        bar_c = "#" * int(r["collision_py"] * 30)
        print(f"    c={r['c']:>2}  raw gap ratio {max(r['gap'] / gap1, 0):>5.2f} {bar_g:<31} "
              f"collide {r['collision_py'] * 100:>5.1f}% {bar_c:<24} "
              f"noise-recovery gap {gnc:>6.3f}")
    corr = float(np.corrcoef(coll, gap_nc)[0, 1])
    print(f"    corr(collision, noise-recovery gap) = {corr:.3f}  "
          f"(strongly negative = collision destroys exactly what abduction used)")

    # ---- D2: what the residual/coarse 'gap' actually is ----------------------------
    print(f"\n[D2] Decomposition — is the CF advantage noise RECOVERY or factual-COPYING?")
    print(f"     error_COPY = decode the OBSERVED token under a' (no model, no abduction).")
    print(f"     When CF ~= COPY and 'CF==obs' -> 100%, abduction has degenerated to")
    print(f"     repeating the factual outcome (Gumbel stability prior), NOT noise recovery:")
    print(f"     {'c':>3} | {'error_CF':>8} | {'error_COPY':>10} | {'error_IV':>8} | "
          f"{'CF==obs':>7}")
    for r in rows:
        print(f"     {r['c']:>3} | {r['error_CF']:>8.3f} | {r['error_COPY']:>10.3f} | "
              f"{r['error_IV']:>8.3f} | {r['cf_repeats_obs'] * 100:>6.1f}%")

    # ---- D3: gap split by per-tuple collision status (free tuples) ------------------
    print(f"\n[D3] Free-tuple gap split by collision status (collided tuples carry ZERO")
    print(f"     noise information -> any gap there is the stability prior, not abduction;")
    print(f"     level-dependent subsets — a diagnostic, not the paired headline):")
    print(f"     {'c':>3} | {'n_noncoll':>9} | {'gap (non-collided)':>18} | "
          f"{'n_coll':>6} | {'gap (collided)':>14}")
    for r in rows:
        t = r["tuples"]
        e_cf = np.array(t["e_cf"]); e_iv = np.array(t["e_iv"])
        fl = np.array(t["collide"])
        nc, cl = fl == 0, fl == 1
        g_nc = float(e_iv[nc].mean() - e_cf[nc].mean()) if nc.any() else float("nan")
        g_cl = float(e_iv[cl].mean() - e_cf[cl].mean()) if cl.any() else float("nan")
        print(f"     {r['c']:>3} | {int(nc.sum()):>9} | {g_nc:>18.3f} | "
              f"{int(cl.sum()):>6} | {g_cl:>14.3f}")

    # ---- E: training sanity --------------------------------------------------------
    # The check that matters: at the levels where the gap COLLAPSED, accuracy must not
    # have cratered (else the collapse could be underfitting, not information loss).
    accs = [r["top1_player"] for r in rows]
    collapsed = [r for r in rows if r["gap"] / gap1 < 0.5]
    ok_E = all(r["top1_player"] >= accs[0] - 0.05 for r in collapsed)
    print(f"\n[E] Training sanity: at every COLLAPSED level, held-out player_y top-1")
    print(f"    must not crater below the c=1 level (coarser tokens are easier):")
    print(f"    all levels: {['%.1f%%' % (a * 100) for a in accs]}")
    print(f"    collapsed levels ({[r['c'] for r in collapsed]}): "
          f"{['%.1f%%' % (r['top1_player'] * 100) for r in collapsed]} vs c=1 "
          f"{accs[0] * 100:.1f}%  -> {'OK — collapse is not a training artifact' if ok_E else '** SUSPECT: retrain before reporting **'}")

    # ---- secondary axis --------------------------------------------------------------
    nh = None
    if os.path.exists(RESULTS_NOHIST):
        nh = load(RESULTS_NOHIST)["rows"]
        print(f"\n[secondary] history OFF (N=1) vs ON (N=8), gap by level:")
        print(f"    {'c':>3} | {'gap (hist ON)':>13} | {'gap (hist OFF)':>14}")
        for r, q in zip(rows, nh):
            print(f"    {r['c']:>3} | {r['gap']:>13.3f} | {q['gap']:>14.3f}")

    # ---- plot ------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cs = [r["c"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(cs, [max(r["gap"] / gap1, 0) for r in rows], "o-",
                label="gap(c) / gap(1)  (CF advantage retained)")
        ax.plot(cs, coll, "s--", label="stuck/free outcome-token collision rate")
        if knee:
            ax.axvline(knee, color="gray", ls=":", lw=1)
            ax.annotate(f"knee c*={knee}px", (knee, 0.52), fontsize=9, color="gray")
        rl = rows[-1]
        if rl["gap"] / gap1 > 0.5 and rl["collision_py"] > 0.5:
            ax.annotate("copying artifact,\nnot noise recovery (D2)",
                        (rl["c"], rl["gap"] / gap1), fontsize=8, color="tab:blue",
                        xytext=(rl["c"] * 0.45, rl["gap"] / gap1 + 0.12),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))
        ax.axvline(paddle_step, color="tab:red", ls=":", lw=1)
        ax.annotate(f"paddle step ≈{paddle_step:.1f}px", (paddle_step, 0.9),
                    fontsize=9, color="tab:red")
        ax.set_xscale("log", base=2)
        ax.set_xticks(cs); ax.set_xticklabels(cs)
        ax.set_xlabel("coarsening bin width c (px)")
        ax.set_ylabel("fraction")
        ax.set_title(f"Rung 2.5: CF advantage vs coarseness "
                     f"(s={data['s']}, fs={data['frameskip']})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(PLOT, dpi=150)
        print(f"\n[plot] -> {PLOT}")
    except ImportError:
        print("\n[plot] matplotlib unavailable -- ASCII bars above are the figure")

    # ---- the spec paragraph -----------------------------------------------------------
    kp = knee if knee else max(r["c"] for r in rows)
    last = rows[-1]
    print("\n" + "=" * 78)
    print("RUNG 3 SPEC:")
    print(f"Gumbel-Max abduction retains >=50% of its counterfactual advantage only")
    print(f"while the representation resolves paddle/ball deltas finer than {kp} px")
    print(f"(the fs=2 free-step paddle delta is ~{paddle_step:.1f} px). A Rung 3 tokenizer")
    print(f"must preserve at least this resolution -- and the stuck-vs-non-stuck outcome")
    print(f"distinction in particular (collision rate at the knee: "
          f"{next(r['collision_py'] for r in rows if r['c'] == kp) * 100:.0f}%) -- or")
    print(f"counterfactuals will degenerate to interventions. Beyond the knee any")
    print(f"apparent CF-vs-IV gap is NOT noise recovery but the Gumbel stability prior")
    print(f"copying the factual outcome (at c={last['c']}: CF==obs "
          f"{last['cf_repeats_obs'] * 100:.0f}% of the time, error_CF "
          f"{last['error_CF']:.2f} vs error_COPY {last['error_COPY']:.2f}) -- and on")
    print(f"the residual non-collided steps abduction is actively HARMFUL "
          f"(noise-recovery gap {gap_nc[-1]:.2f} at c={last['c']}): a too-coarse")
    print(f"tokenizer does not gracefully lose the CF advantage, it inverts it.")
    if nh:
        print(f"History axis: with history OFF the c=1 gap is "
              f"{nh[0]['gap']:.3f} vs {gap1:.3f} with history ON; see table above for "
              f"whether history rescues coarse levels.")
    print("=" * 78)


if __name__ == "__main__":
    main()
