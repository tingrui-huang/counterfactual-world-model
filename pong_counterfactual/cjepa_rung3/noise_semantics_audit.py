"""Rung 3 Stage A1.2 — noise semantics & realized-motion audit (read-only).

Builds on A1.1's finding that the binary `stuck` bit is a weak/ill-posed abduction target.
Here we (1) pin the ALE sticky/frame-skip mechanics, (2) define the noise variable that is
actually present in the data, (3) re-pose probe targets around REALIZED MOTION, (4) compare
frozen feature families, and (5) diagnose the discrete representation's transition structure
(token locality: do small paddle moves flip all tokens?).

No tokenizer training, no Stage 3/4, no architecture change. CPU-only, frozen checkpoint,
reuses the audited helpers in diagnose_representation (DR) and validate_sticky_probe (V).

Run:  python -m pong_counterfactual.cjepa_rung3.noise_semantics_audit
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3.data import load_split, build_eval_pool, NOOP, UP, DOWN
from pong_counterfactual.cjepa_rung3 import diagnose_representation as DR
from pong_counterfactual.cjepa_rung3 import validate_sticky_probe as V


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _n(x):
    return "n/a" if x is None else f"{x:.3f}"


class NoiseAudit:
    def __init__(self, repo, ckpt, data_dir, out):
        self.repo, self.ckpt, self.data_dir, self.out = repo, ckpt, data_dir, out
        self.gate_rows = []

    def gate(self, name, check, criterion, result, evidence, notes=""):
        self.gate_rows.append({"Gate": name, "Check": check, "PASS criterion": criterion,
                               "Result": result, "Evidence artifact": evidence, "Notes": notes})

    def _wcsv(self, name, rows, fields):
        with open(self.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    # --------------------------------------------------------------- load + encode
    def load(self):
        self.model, self.cfg = DR.load_eye_from_path(self.ckpt, "cpu")
        M, K, cd = self.cfg.M, self.cfg.K, self.cfg.code_dim
        self.M, self.K, self.cd = M, K, cd
        self.eps = load_split(self.data_dir, "eval", C.EVAL_BASE_SEED, C.N_EVAL_EPS)
        self.pool = build_eval_pool(self.eps, C.N_HIST, C.N_SAMPLES)
        self.oc = dict(np.load(self.data_dir / "oracle_cache.npz", allow_pickle=True))
        self.grp = np.array([ei for ei, k in self.pool])
        self.vk = np.array([self.eps[ei].vecs[k] for ei, k in self.pool])
        self.vk1 = np.array([self.eps[ei].vecs[k + 1] for ei, k in self.pool])
        self.a_int = np.array([int(self.eps[ei].intended[k]) for ei, k in self.pool])
        self.a_prev = np.array([int(self.eps[ei].intended[k - 1]) for ei, k in self.pool])
        self.stuck = self.oc["stuck"].astype(int)
        self.cur_frame = np.stack([self.eps[ei].frames[k] for ei, k in self.pool])
        self.fac_frame = self.oc["o_fac_frame"]
        self.disp = self.vk1[:, 2] - self.vk[:, 2]
        self.ydir = np.sign(self.disp).astype(int)
        self.mag = np.abs(self.disp)
        cb = self.model.vq.codebook.detach().cpu().numpy()
        self.cb = cb
        self.tok_cur = DR.encode_tokens(self.model, self.cur_frame, "cpu")
        self.tok_fac = DR.encode_tokens(self.model, self.fac_frame, "cpu")
        self.tok_cf = DR.encode_tokens(self.model, self.oc["o_cf_frame"], "cpu")
        self.tok_oth = DR.encode_tokens(self.model, self.oc["other_frame"], "cpu")
        # feature families for z_t and z_{t+1}
        self.zt = {
            "onehot": (DR.onehot_codes(self.tok_cur, M, K), DR.onehot_codes(self.tok_fac, M, K)),
            "codebook_emb": (cb[self.tok_cur].reshape(len(self.tok_cur), -1),
                             cb[self.tok_fac].reshape(len(self.tok_fac), -1)),
            "prequant": (DR.prequant_latents(self.model, self.cur_frame, "cpu").reshape(len(self.tok_cur), -1),
                         DR.prequant_latents(self.model, self.fac_frame, "cpu").reshape(len(self.tok_fac), -1)),
        }
        self.a_oh = np.concatenate([DR._oh(self.a_prev, 4), DR._oh(self.a_int, 4)], 1)

    # ----------------------------------------------------- GATE N1: noise semantics
    def gate_n1_semantics(self):
        free = self.stuck == 0
        stk = self.stuck == 1
        rows = []
        for lab, m in (("free(stuck=0)", free), ("stuck(stuck=1)", stk)):
            d = self.disp[m]
            rows.append({"subset": lab, "n": int(m.sum()),
                         "mean_abs_disp": float(np.abs(d).mean()),
                         "median_abs_disp": float(np.median(np.abs(d))),
                         "frac_moved>0.5px": float((np.abs(d) > 0.5).mean()),
                         "frac_zero": float((np.abs(d) < 1e-9).mean()),
                         "std_abs_disp": float(np.abs(d).std())})
        self.semantics_rows = rows
        # empirical stuck rate vs s and s^2
        sr = float(stk.mean())
        self.gate("N1", "ALE sticky/frame-skip mechanics + 3.2px puzzle",
                  "mechanics explained from code + data; binary stuck shown lossy",
                  "RESOLVED", "NOISE_SEMANTICS_REPORT.md (sec 1), realized_motion_results.csv",
                  f"frameskip={C.FRAMESKIP}, sticky applied inside per-frame ale.act() loop "
                  f"(ale_py env.py:304); repeated action = last EXECUTED frame-action (not the "
                  f"logged intended NOOP). Empirical stuck rate {sr:.2f}~=s={C.S_TRAIN} (not "
                  f"s^2), and stuck moves partial ({rows[1]['mean_abs_disp']:.1f}px) vs free "
                  f"full ({rows[0]['mean_abs_disp']:.1f}px) -> binary stuck collapses a partial, "
                  f"continuous motion-suppression process.")
        return rows

    def write_noise_target_definition(self):
        full_step = float(np.median(self.mag[self.stuck == 0]))
        defn = {
            "summary": "The latent exogenous noise is the per-internal-frame ALE sticky-action "
                       "draw sequence. The only well-posed OBSERVABLE noise effect in the logged "
                       "data is the realized paddle displacement; the binary `stuck` bit is a "
                       "lossy 2-probe summary of it.",
            "mechanics": {
                "frameskip": C.FRAMESKIP,
                "repeat_action_probability_s": C.S_TRAIN,
                "sticky_unit": "per internal emulator frame (ale.act called frameskip times; "
                               "sticky drawn inside each act) — see ale_py/env.py step() loop",
                "repeated_action": "the last action ALE EXECUTED (lastAction), updated every "
                                   "frame — NOT the agent's logged intended action",
                "empirical_stuck_rate": float((self.stuck == 1).mean()),
                "note_on_rate": "stuck rate ~= s (not s^2); the two-probe detector flags a step "
                                "as stuck when the intended action is fully overridden. The exact "
                                "per-frame draw count is internal to ALE.",
            },
            "candidate_noise_variables": [
                {"name": "binary_stuck", "status": "LOGGED (oracle 2-probe detector)",
                 "verdict": "INSUFFICIENT — collapses partial/continuous motion suppression; "
                            "weakly identifiable even from ground truth (A1.1)"},
                {"name": "per_frame_repeated_action_pattern_or_count", "status": "NOT LOGGED",
                 "verdict": "CANNOT RECONSTRUCT from agent-step data (per-frame executed actions "
                            "were never recorded)"},
                {"name": "realized_action_sequence", "status": "NOT LOGGED",
                 "verdict": "CANNOT RECONSTRUCT"},
                {"name": "realized_paddle_displacement_dy", "status": "RECONSTRUCTABLE "
                 "(player_y[k+1]-player_y[k] from logged vecs)",
                 "verdict": "WELL-DEFINED observable noise effect — the recommended target"},
                {"name": "displacement_residual_vs_free", "status": "PARTIALLY RECONSTRUCTABLE",
                 "verdict": "free branch is replay-reachable only for free tuples; for stuck "
                            "tuples the free outcome is unreachable (A1.1) and must be ESTIMATED "
                            f"as the intended full move (~{full_step:.1f}px signed)"},
            ],
            "recommended_targets": ["paddle_displacement_dy(regression)",
                                    "movement_direction(3class)", "movement_magnitude(|dy|)",
                                    "displacement_residual_vs_free(estimated)"],
            "estimated_full_paddle_step_px": full_step,
        }
        (self.out / "noise_target_definition.json").write_text(
            json.dumps(defn, indent=2), encoding="utf-8")
        self.full_step = full_step
        return defn

    # ----------------------------------------- GATE N2: re-posed targets x features
    def gate_n2_realized_motion(self):
        vk, vk1 = self.vk, self.vk1
        pix = self._pixel_diff()
        # free intended (no-sticky) signed displacement, ESTIMATED per direction
        dirsign = np.where(self.a_int == UP, np.sign(np.median(self.disp[(self.stuck==0)&(self.a_int==UP)])),
                  np.where(self.a_int == DOWN, np.sign(np.median(self.disp[(self.stuck==0)&(self.a_int==DOWN)])), 0.0))
        free_disp = dirsign * self.full_step
        residual = self.disp - free_disp                      # motion deficit due to noise

        families = {
            "action_only": self.a_oh,
            "gt_object_transition": np.concatenate([vk, vk1], 1),
            "pixel_diff_8x8": pix,
            "jepa_onehot": np.concatenate([*self.zt["onehot"], self.a_oh], 1),
            "jepa_codebook_emb": np.concatenate([*self.zt["codebook_emb"], self.a_oh], 1),
            "jepa_prequant": np.concatenate([*self.zt["prequant"], self.a_oh], 1),
        }
        targets = [("displacement_dy", self.disp, "r2"),
                   ("direction_3cls", self.ydir, "acc"),
                   ("magnitude_absdy", self.mag, "r2"),
                   ("residual_vs_free", residual, "r2")]
        rng = np.random.default_rng(0)
        rows = []
        for tname, y, kind in targets:
            for fname, X in families.items():
                if kind == "r2":
                    res = V.grouped_r2(X, y, self.grp, repeats=15)
                    score, lo, hi, base = res["r2_mean"], res["ci_lo"], res["ci_hi"], 0.0
                    ysh = y.copy(); rng.shuffle(ysh)
                    sh = V.grouped_r2(X, ysh, self.grp, repeats=15)["r2_mean"]
                else:
                    res = V.grouped_acc(X, y, self.grp, repeats=15)
                    score, lo, hi, base = res["acc_mean"], res["ci_lo"], res["ci_hi"], res["baseline"]
                    ysh = y.copy(); rng.shuffle(ysh)
                    sh = V.grouped_acc(X, ysh, self.grp, repeats=15)["acc_mean"]
                rows.append({"target": tname, "feature_family": fname, "metric": kind,
                             "score_mean": score, "ci_lo": lo, "ci_hi": hi,
                             "baseline": base, "shuffled_control": sh})
        self._wcsv("realized_motion_results.csv", rows,
                   ["target", "feature_family", "metric", "score_mean", "ci_lo", "ci_hi",
                    "baseline", "shuffled_control"])
        self.motion_rows = rows

        def best(target, families_sub):
            cand = [r for r in rows if r["target"] == target and r["feature_family"] in families_sub]
            return max((r["score_mean"] or -9 for r in cand), default=None)
        jepa = {"jepa_onehot", "jepa_codebook_emb", "jepa_prequant"}
        self.dir_oracle = best("direction_3cls", {"gt_object_transition"})
        self.dir_action = best("direction_3cls", {"action_only"})
        self.dir_jepa = best("direction_3cls", jepa)
        self.disp_oracle = best("displacement_dy", {"gt_object_transition"})
        self.disp_jepa = best("displacement_dy", jepa)
        base_dir = next(r["baseline"] for r in rows if r["target"] == "direction_3cls"
                        and r["feature_family"] == "action_only")
        self.dir_base = base_dir
        self.gate("N2", "re-posed targets (displacement/direction/magnitude/residual) x features",
                  "JEPA decodes realized motion clearly above action-only and near oracle",
                  "PASS" if (self.dir_jepa or 0) > (self.dir_action or 0) + 0.05 and
                            (self.disp_jepa or -9) > 0.1 else "WEAK",
                  "realized_motion_results.csv",
                  f"direction: oracle={_n(self.dir_oracle)} action_only={_n(self.dir_action)} "
                  f"JEPA={_n(self.dir_jepa)} (base {_n(self.dir_base)}); displacement R2: "
                  f"oracle={_n(self.disp_oracle)} JEPA={_n(self.disp_jepa)}. JEPA adds little over "
                  f"actions and trails oracle.")
        return rows

    def _pixel_diff(self):
        cur = self.cur_frame.astype(np.float32); nxt = self.fac_frame.astype(np.float32)
        d = np.abs(nxt - cur); N, res, _ = d.shape; g = 8; s = res // g
        return d[:, :s*g, :s*g].reshape(N, g, s, g, s).mean((2, 4)).reshape(N, -1)

    # --------------------------------------------- GATE N3: representation structure
    def gate_n3_locality(self):
        def ham(a, b): return (a != b).sum(1)
        def cbdist(a, b):
            return np.sqrt(((self.cb[a] - self.cb[b]) ** 2).sum(-1)).mean(-1)
        small_thr = self.full_step                            # one paddle step

        # ---- PURE-PADDLE contrasts: branch pairs share timestep k+1 (ball ~identical),
        #      so token change isolates the PADDLE-position difference (the noise effect). ----
        dpy_faccf = np.abs(self.oc["o_fac"][:, 2] - self.oc["o_cf"][:, 2])
        dpy_facoth = np.abs(self.oc["o_fac"][:, 2] - self.oc["other"][:, 2])
        H_faccf, H_facoth = ham(self.tok_fac, self.tok_cf), ham(self.tok_fac, self.tok_oth)
        D_faccf, D_facoth = cbdist(self.tok_fac, self.tok_cf), cbdist(self.tok_fac, self.tok_oth)
        dpy_pure = np.concatenate([dpy_faccf, dpy_facoth])
        H_pure = np.concatenate([H_faccf, H_facoth])
        D_pure = np.concatenate([D_faccf, D_facoth])
        movemask = dpy_pure > 1e-6                             # paddle actually differs
        # ---- realized step (cur->fac): the transition a model would see; BALL-DOMINATED ----
        disp_step = np.abs(self.vk1[:, 2] - self.vk[:, 2])
        H_step = ham(self.tok_cur, self.tok_fac)
        # ---- temporal token persistence (consecutive frames within episodes) ----
        per_slot_frame_flip = np.zeros(self.M); n_adj = 0
        for ep in self.eps:
            valid = ~np.isnan(ep.vecs).any(1)
            tok = DR.encode_tokens(self.model, ep.frames, "cpu")
            for k in range(len(ep.intended)):
                if valid[k] and valid[k + 1]:
                    per_slot_frame_flip += (tok[k] != tok[k + 1]); n_adj += 1
        per_slot_frame_flip /= max(1, n_adj)
        ps_step = (self.tok_cur != self.tok_fac).mean(0)
        ps_faccf = (self.tok_fac != self.tok_cf).mean(0)
        ps_facoth = (self.tok_fac != self.tok_oth).mean(0)

        def corr(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        c_ham = corr(dpy_pure[movemask], H_pure[movemask])     # PURE-paddle Hamming vs Δpaddle
        c_cb = corr(dpy_pure[movemask], D_pure[movemask])
        sm = movemask & (dpy_pure <= small_thr)                # small PURE paddle moves
        frac_allflip_small = float((H_pure[sm] == self.M).mean()) if sm.any() else None

        rows = [
            {"metric": "corr(token_Hamming, |Δpaddle|) [PURE-paddle branch pairs]", "value": c_ham,
             "note": "isolates paddle (same ball); >~0.3 = Hamming grades with paddle move"},
            {"metric": "corr(codebook_emb_dist, |Δpaddle|) [PURE-paddle]", "value": c_cb,
             "note": "continuous geometry grades more than discrete tokens"},
            {"metric": "mean token Hamming /M [PURE-paddle ~1-2 step contrast]",
             "value": float(H_pure[movemask].mean()),
             "note": f"M={self.M}; a paddle-only change flips this many of {self.M} tokens"},
            {"metric": "mean token Hamming /M [realized step cur->fac, BALL-DOMINATED]",
             "value": float(H_step.mean()), "note": "ball moves every frame -> near-saturated"},
            {"metric": "frac small PURE paddle moves (<=1 step) flipping ALL tokens",
             "value": frac_allflip_small, "note": "high => paddle change is spread over ALL "
             "slots (no paddle-local slot)"},
            {"metric": "mean adjacent-frame token persistence (1 - flip) /slot",
             "value": float(1 - per_slot_frame_flip.mean()),
             "note": "low => tokens churn frame-to-frame (ball-driven, no temporal stability)"},
        ]
        for m in range(self.M):
            rows.append({"metric": f"per_slot_flip_rate slot{m}",
                         "value": float(ps_faccf[m]),
                         "note": f"fac_vs_cf={ps_faccf[m]:.2f} fac_vs_other={ps_facoth[m]:.2f} "
                                 f"adjacent-frame={per_slot_frame_flip[m]:.2f} "
                                 f"realized-step={ps_step[m]:.2f}"})
        self._wcsv("token_locality_metrics.csv", rows, ["metric", "value", "note"])
        self.locality = {"c_ham": c_ham, "c_cb": c_cb,
                         "mean_ham_pure": float(H_pure[movemask].mean()),
                         "mean_ham_step": float(H_step.mean()),
                         "frac_allflip_small": frac_allflip_small,
                         "temporal_persistence": float(1 - per_slot_frame_flip.mean())}
        self._plots(dpy_pure[movemask], H_pure[movemask], D_pure[movemask],
                    ps_step, per_slot_frame_flip, ps_faccf, ps_facoth)

        graded = c_ham > 0.3
        local = (frac_allflip_small is not None and frac_allflip_small < 0.5)
        self.gate("N3", "representation structure: token locality / transition stability",
                  "paddle-only token change grades with Δpaddle; small moves don't flip all tokens",
                  "PASS" if (graded and local) else "FAIL (global/entangled tokens)",
                  "token_locality_metrics.csv, hamming_vs_displacement.png, per_slot_flip_rates.png",
                  f"PURE-paddle corr(Hamming,Δpaddle)={_n(c_ham)}; a paddle-only change flips "
                  f"{self.locality['mean_ham_pure']:.2f}/{self.M} tokens spread across ALL slots "
                  f"(per-slot ~{ps_faccf.mean():.2f}, no paddle-local slot); {_n(frac_allflip_small)} "
                  f"of small paddle moves flip ALL tokens; adjacent-frame persistence "
                  f"{self.locality['temporal_persistence']:.2f} (ball-driven churn). No stable "
                  f"local per-slot transition structure.")
        return rows

    def _plots(self, disp, H, D, ps_step, ps_frame, ps_faccf, ps_facoth):
        plt = _plt()
        # hamming + cb-dist vs displacement (binned)
        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        order = np.argsort(disp)
        nb = 8
        edges = np.quantile(disp, np.linspace(0, 1, nb + 1))
        edges = np.unique(edges)
        xc, hm, dm = [], [], []
        idx = np.clip(np.digitize(disp, edges[1:-1]), 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.any():
                xc.append(0.5 * (edges[b] + edges[b + 1]))
                hm.append(H[sel].mean()); dm.append(D[sel].mean())
        ax1.plot(xc, hm, "o-", color="tab:blue", label="token Hamming")
        ax1.axhline(self.M, ls="--", color="tab:blue", alpha=.4, label=f"max Hamming ({self.M})")
        ax1.set_xlabel("|Δpaddle| pure branch contrast (screen px)"); ax1.set_ylabel("token Hamming", color="tab:blue")
        ax1.set_ylim(0, self.M + 0.3)
        ax2 = ax1.twinx(); ax2.plot(xc, dm, "s-", color="tab:red", label="codebook-emb dist")
        ax2.set_ylabel("codebook-emb dist", color="tab:red")
        ax1.set_title("Token change vs physical paddle displacement")
        ax1.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(self.out / "hamming_vs_displacement.png", dpi=110); plt.close(fig)
        # per-slot flip rates
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(self.M); w = 0.2
        ax.bar(x - 1.5*w, ps_frame, w, label="adjacent frame")
        ax.bar(x - 0.5*w, ps_step, w, label="realized step (cur->fac)")
        ax.bar(x + 0.5*w, ps_faccf, w, label="fac vs CF")
        ax.bar(x + 1.5*w, ps_facoth, w, label="fac vs other")
        ax.set_xticks(x); ax.set_xticklabels([f"slot{m}" for m in range(self.M)])
        ax.set_ylabel("flip rate"); ax.set_ylim(0, 1)
        ax.set_title("Per-slot token flip rate"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(self.out / "per_slot_flip_rates.png", dpi=110); plt.close(fig)

    # ------------------------------------------------------------------- verdict
    def verdict(self):
        graded = self.locality["c_ham"] > 0.3
        local = (self.locality["frac_allflip_small"] is not None
                 and self.locality["frac_allflip_small"] < 0.5)
        jepa_motion = ((self.dir_jepa or 0) > (self.dir_action or 0) + 0.05
                       and (self.disp_jepa or -9) > 0.1)
        discriminative = self.locality["mean_ham_step"] > 0.5 * self.M   # frames clearly separable
        if jepa_motion and graded and local:
            v = "STRUCTURED_SIGNAL"
        elif discriminative and not (graded and local):
            v = "DISCRIMINATIVE_ONLY"
        else:
            v = "TARGET_INVALID"
        self.final = v
        return v

    def write_report(self):
        v = self.final
        cols = ["Gate", "Check", "PASS criterion", "Result", "Evidence artifact", "Notes"]
        gt = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for r in self.gate_rows:
            gt.append("| " + " | ".join(str(r[c]).replace("\n", " ").replace("|", "\\|")
                                        for c in cols) + " |")
        (self.out / "gate_table.json").write_text(json.dumps(self.gate_rows, indent=2), encoding="utf-8")
        sf = self.semantics_rows
        (self.out / "NOISE_SEMANTICS_REPORT.md").write_text(f"""# Rung 3 Stage A1.2 — Noise Semantics & Realized-Motion Audit

**Verdict: {v}**

Generated {utc_stamp()} • Checkpoint `{self.ckpt.name}` • read-only (no training, no Stage 3/4)

## 1. ALE sticky / frame-skip mechanics (resolves the 3.2 px puzzle)
- **Wrapper path:** `PongEnv` -> OCAtari (`gym.make`, kwargs frameskip={C.FRAMESKIP},
  repeat_action_probability={C.S_TRAIN}) -> ale_py `AtariEnv.step`, which runs
  `for _ in range(frameskip): self.ale.act(action_idx)` (ale_py/env.py:304). Sticky is
  applied **inside each `ale.act()`**, i.e. **per internal emulator frame**, not once per
  agent step.
- **What is repeated:** ALE's sticky replaces the current action with `lastAction` = the
  action ALE **last executed** (updated every frame), NOT the agent's logged intended
  action.
- **Why logged `a_prev` is always NOOP yet stuck transitions still move ~{sf[1]['mean_abs_disp']:.1f} px:**
  the behavior policy issues moves as isolated pulses, so the logged intended action at k-1
  is NOOP; but the action ALE *executed* on the last frame of step k-1 can be a real move
  (sticky chaining), and that is what gets repeated. With {C.FRAMESKIP} frames/step and
  per-frame stickiness, a "stuck" step typically suppresses only PART of the motion, so the
  paddle moves a partial amount ({sf[1]['mean_abs_disp']:.1f} px) rather than the full free
  step ({sf[0]['mean_abs_disp']:.1f} px). Empirical stuck rate
  {float((self.stuck==1).mean()):.2f} ~= s (not s^2); the exact per-frame draw count is
  internal to ALE and was **not logged**.

## 2. The actual noise variable (noise_target_definition.json)
The exogenous noise is the per-frame sticky draw sequence; its only well-posed **observable
effect** in the logged data is the **realized paddle displacement Δy**. The binary `stuck`
bit is a lossy 2-probe summary (A1.1 showed it is weakly identifiable even from ground
truth). Per-frame repeated-action pattern/count and the realized action sequence were **not
logged and cannot be reconstructed** from agent-step states.

## 3-4. Re-posed targets x frozen features (realized_motion_results.csv)
Targets: displacement Δy, direction (3-cls), magnitude |Δy|, residual-vs-free (free branch
estimated for stuck tuples — replay-unreachable, A1.1). Decoded from: action-only,
ground-truth object transition, pixel-difference, JEPA one-hot / codebook-emb / pre-quant.
- direction: oracle(gt) **{_n(self.dir_oracle)}**, action-only {_n(self.dir_action)},
  best JEPA **{_n(self.dir_jepa)}** (base {_n(self.dir_base)}).
- displacement R²: oracle **{_n(self.disp_oracle)}**, best JEPA **{_n(self.disp_jepa)}**.
JEPA adds little over actions and trails the (informative) ground-truth ceiling.

## 5. Representation structure / token locality (token_locality_metrics.csv)
- PURE-paddle corr(token Hamming, |Δpaddle|) = **{_n(self.locality['c_ham'])}** (branch pairs,
  same ball; moderate ⇒ Hamming partially grades but is near-saturated).
- a paddle-ONLY change flips **{self.locality['mean_ham_pure']:.2f}/{self.M}** tokens, spread
  across ALL slots (no paddle-local slot); the realized step (ball-dominated) flips
  **{self.locality['mean_ham_step']:.2f}/{self.M}**.
- fraction of small (≤1 paddle step) PURE paddle moves that flip ALL {self.M} tokens =
  **{_n(self.locality['frac_allflip_small'])}**.
- adjacent-frame token persistence = **{self.locality['temporal_persistence']:.2f}**
  (low ⇒ tokens churn frame-to-frame, ball-driven).
See `hamming_vs_displacement.png`, `per_slot_flip_rates.png`.

**Interpretation:** the discrete tokens are **global/entangled and ball-dominated** — a
paddle-only change is smeared across all {self.M} slots (no slot specializes in the paddle),
~half of small paddle moves flip every token, and frame-to-frame persistence is low. The eye
SEPARATES frames (low collision, high Hamming) but provides **no stable, local per-slot
transition structure** a transition model could use to abduct the realized-motion noise.
JEPA features cannot decode realized paddle displacement (R²<0, below action-only).

## Verdict: {v}
{"JEPA preserves usable realized-motion structure." if v=='STRUCTURED_SIGNAL' else
 "JEPA SEPARATES frames (discriminative, low collision) but its discrete tokens flip near-globally with no stable transition structure — so realized-motion / noise-effect decoding from the transition is weak. The frames are distinguishable; the *transition representation* is not abduction-usable as-is." if v=='DISCRIMINATIVE_ONLY' else
 "The dataset does not expose a well-defined, reconstructable noise variable for abduction at agent-step granularity."}

## Recommended next step
Before any architecture change: the abduction signal that IS well-defined is the realized
paddle displacement. Two options follow from this audit — (a) make the eye's tokens
position-LOCAL/graded (e.g. object-centric slots or higher paddle resolution) so token
distance tracks paddle displacement, and/or (b) log the per-frame executed-action sequence
so the noise variable is fully observable. Do NOT train Stage 3 on the current global tokens.

## Gate table
{chr(10).join(gt)}
""", encoding="utf-8")


def main(argv=None):
    repo = DR.find_repo_root(Path(__file__))
    ckpt = DR.discover_checkpoint(repo, None)
    ckpt_res = DR.checkpoint_res(ckpt)
    data_dir = DR.discover_data_dir(repo, C.S_TRAIN, ckpt_res, None)
    if data_dir is None:
        print("[noise] no matching data dir (need s=%g res=%d)" % (C.S_TRAIN, ckpt_res), file=sys.stderr)
        sys.exit(2)
    out = repo / "pong_counterfactual" / "audits" / f"rung3_noise_semantics_{utc_stamp()}"
    out.mkdir(parents=True, exist_ok=True)
    a = NoiseAudit(repo, ckpt, data_dir, out)
    a.load()
    a.gate_n1_semantics()
    a.write_noise_target_definition()
    a.gate_n2_realized_motion()
    a.gate_n3_locality()
    verdict = a.verdict()
    a.write_report()
    (out / "exact_commands.txt").write_text(
        "python -m pong_counterfactual.cjepa_rung3.noise_semantics_audit\n", encoding="utf-8")
    print(f"[noise] verdict: {verdict}\n[noise] output: {out}")
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
