"""End-to-end reconstruction-AE diagnostic: gates D0, A0, A1, P1, P2, P3.

Compares three representations on ONE aligned eval pool + episode-grouped split:
  * object-state    : ground-truth 4-vector (oracle reference / ceiling).
  * discrete-JEPA   : frozen tokenizer_M4_K64_res84 one-hot tokens (downstream form)
                      and, as a JEPA best-case, its continuous pre-quant latent.
  * reconstruction  : frozen conv-AE spatial latent trained here (Step 2).

Everything read-only w.r.t. frozen artifacts. Run:
  python -m pong_counterfactual.cjepa_rung3_diag_ae.run_diagnostic \
      --out pong_counterfactual/cjepa_rung3_diag_ae/outputs/main [--steps 4000]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from pong_counterfactual.cjepa_rung3_diag_ae import common as CM
from pong_counterfactual.cjepa_rung3_diag_ae import train_ae as TR
from pong_counterfactual.cjepa_rung3_diag_ae.ae_model import ConvAutoencoder, AEConfig
from pong_counterfactual.cjepa_rung3_diag_ae import probes as PB

# frozen-JEPA helpers reused verbatim from the existing audit (alignment guarantee)
from pong_counterfactual.cjepa_rung3.diagnose_representation import (
    load_eye_from_path, encode_tokens, prequant_latents, onehot_codes,
)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def sprint(*a):
    """Console print that survives a non-UTF8 (e.g. gbk) Windows console — files are
    always written UTF-8, only the mirror-to-console is sanitized."""
    s = " ".join(str(x) for x in a)
    try:
        print(s)
    except UnicodeEncodeError:
        import sys
        print(s.encode(sys.stdout.encoding or "ascii", "replace").decode(
            sys.stdout.encoding or "ascii"))


# ============================================================ representations
class Reps:
    """Builds the aligned feature matrices for every representation, on the eval pool."""

    def __init__(self, bundle: CM.EvalBundle, ae_model, eye, device="cpu"):
        self.b = bundle
        self.device = device
        # frozen encodings of the three frame sets we need
        self._ae = ae_model
        self._eye = eye
        self.M = eye.cfg.M
        self.K = eye.cfg.K

    # ---- single-frame features (for P1) --------------------------------------
    def object_state_1f(self):
        return self.b.vec_k.astype(np.float64)                 # (N,4) oracle ceiling

    def jepa_tokens_1f(self, frames):
        return onehot_codes(encode_tokens(self._eye, frames, self.device), self.M, self.K)

    def jepa_prequant_1f(self, frames):
        z = prequant_latents(self._eye, frames, self.device)   # (N,M,cd)
        return z.reshape(len(frames), -1)

    def ae_1f(self, frames):
        return self._ae.encode_numpy(frames, self.device)      # (N, lat)


REPRESENTATIONS = ("object_state", "jepa_tokens", "jepa_prequant", "ae_recon")


# ============================================================ the driver
class Diagnostic:
    def __init__(self, out: Path, steps: int, device: str, lat_spatial: int):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.steps = steps
        self.device = device
        self.lat_spatial = lat_spatial
        self.gate_rows = []
        self.compare = {}      # rep -> {position, deltay, branch}

    def gate(self, name, result, evidence, blocking="", notes=""):
        self.gate_rows.append({"gate": name, "result": result, "evidence": evidence,
                               "blocking": blocking, "notes": notes})
        print(f"[gate] {name}: {result}  {notes}")

    # -------------------------------------------------- Step 1 / D0
    def step_d0(self):
        b = self.b = CM.load_eval_bundle()
        frames, seeds, py = CM.load_train_frames()
        self.train_frames, self.train_seeds, self.train_py = frames, seeds, py
        tr, te = CM.group_train_test_split(b.ep_seed, 0.3, seed=0)
        self.tr, self.te = tr, te
        overlap = set(b.ep_seed[tr].tolist()) & set(b.ep_seed[te].tolist())
        train_eps = set(seeds.tolist())
        eval_eps = set(b.ep_seed.tolist())
        ep_overlap = train_eps & eval_eps
        # dataset identity
        oracle_npz = b.data_dir / "oracle_cache.npz"
        with np.load(oracle_npz, allow_pickle=True) as z:
            vec_ok = float(z["vec_ok"]); frame_ok = float(z["frame_ok"])
        ex = CM.executed_action(b.a_int, b.a_prev, b.stuck)
        summary = {
            "dataset_dir": str(b.data_dir),
            "oracle_cache_sha256": CM.sha256_file(oracle_npz),
            "n_eval_tuples": len(b.pool),
            "n_train_frames": int(len(frames)),
            "n_train_frames_valid_py": int((~np.isnan(py)).sum()),
            "frame_shape": [b.res, b.res], "frame_dtype": "uint8",
            "res": b.res, "n_eval_episode_groups": int(len(eval_eps)),
            "n_train_episode_groups": int(len(train_eps)),
            "probe_split": {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                            "group_overlap": sorted(overlap)},
            "train_eval_episode_overlap": sorted(ep_overlap),
            "replay_determinism": {"vec_ok": vec_ok, "frame_ok": frame_ok},
            "available_fields": {
                "frame_t": True, "frame_t1": True, "player_y_t": True,
                "player_y_t1": True, "delta_y": True, "intended_action": True,
                "previous_action": True, "sticky_fired": "oracle two-probe label",
                "factual_frame": True, "counterfactual_frame": True,
                "other_branch_frame": True, "episode_id": "episode seed",
                "replay_snapshot": "seed-replay (deterministic)",
            },
            "missing_or_derived_fields": {
                "executed_action": "NOT logged; heuristically = a_prev if stuck else a_int"
                                   " (lossy under ALE per-frame sticky — treated as approx)",
                "sticky_fired_perframe": "internal to ALE, unlogged, unreconstructable",
                "boundary_contact_status": "NOT logged; derived from player_y at extremes",
                "clean_noise_labels": "unlogged; per Rung3 A1.2 audit unreconstructable",
            },
            "delta_y_definition": "player_y_t1 - player_y_t (screen px)",
            "delta_y_range": [float(b.delta_y.min()), float(b.delta_y.max())],
            "player_y_range": [float(b.py_k.min()), float(b.py_k.max())],
            "branch_pairs": {
                "n_valid_fac_cf": int(b.phys["valid_fac_cf"].sum()),
                "n_identical_fac_cf": int(b.phys["identical_fac_cf"].sum()),
                "n_different_fac_cf": int((b.phys["valid_fac_cf"]
                                          & ~b.phys["identical_fac_cf"]).sum()),
            },
            "stuck_counts": {int(k): int(v) for k, v in
                             zip(*np.unique(b.stuck, return_counts=True))},
            "identifiable_count": int(b.identifiable.sum()),
        }
        CM.write_json(self.out / "data_summary.json", summary)
        self._write_data_audit_md(summary)
        # gate D0
        ok = (vec_ok == 1.0 and frame_ok == 1.0 and not overlap and not ep_overlap
              and (~np.isnan(b.py_k)).all() and (~np.isnan(b.py_k1)).all())
        self.gate("D0 Data validity", "PASS" if ok else "FAIL",
                  "data_audit.md, data_summary.json",
                  "" if ok else "recollect data",
                  f"replay {vec_ok*100:.0f}/{frame_ok*100:.0f}%, "
                  f"train/eval episodes disjoint, no split group overlap")
        return ok

    def _write_data_audit_md(self, s):
        lines = [
            "# Data audit (Rung 3 reconstruction-AE diagnostic)\n",
            f"**Dataset:** `{s['dataset_dir']}`  ",
            f"**oracle_cache.npz sha256:** `{s['oracle_cache_sha256'][:16]}…`  ",
            f"**Frames:** {s['n_train_frames']} train "
            f"({s['n_train_frames_valid_py']} valid player_y), "
            f"{s['n_eval_tuples']} fixed eval tuples; shape {s['frame_shape']} "
            f"{s['frame_dtype']}, res {s['res']} (grayscale, crop rows [34,194), "
            "bilinear resize — Rung 3 decision ①).\n",
            "## Primary target\n",
            f"`delta_y = player_y_t1 - player_y_t` (screen px), range "
            f"{s['delta_y_range']}. player_y range {s['player_y_range']}.\n",
            "## Available fields\n",
        ]
        for k, v in s["available_fields"].items():
            lines.append(f"- `{k}`: {v}")
        lines.append("\n## Missing / derived fields (NOT silently invented)\n")
        for k, v in s["missing_or_derived_fields"].items():
            lines.append(f"- `{k}`: {v}")
        lines += [
            "\n## Splits & leakage\n",
            f"- AE trains on TRAIN episodes only ({s['n_train_episode_groups']} groups); "
            f"probes evaluate on EVAL episodes ({s['n_eval_episode_groups']} groups). "
            f"Train/eval episode overlap: {s['train_eval_episode_overlap'] or 'NONE'}.",
            f"- Probe split is episode-grouped ({s['probe_split']['n_train']} train / "
            f"{s['probe_split']['n_test']} test tuples); group overlap: "
            f"{s['probe_split']['group_overlap'] or 'NONE'} → no same-transition leakage.",
            f"- Factual/CF pairs stay grouped per tuple; "
            f"{s['branch_pairs']['n_different_fac_cf']} physically-different vs "
            f"{s['branch_pairs']['n_identical_fac_cf']} identical (stuck) branch pairs.",
            f"- Replay/oracle: seed-replay reproduces factual vec "
            f"{s['replay_determinism']['vec_ok']*100:.0f}% / frame "
            f"{s['replay_determinism']['frame_ok']*100:.0f}%.",
            "\n## Notes / limitations\n",
            "- `sticky_fired`/`stuck` is NOT used as the primary representation target "
            "(weakly identifiable per Rung 3 A1.1 audit). Primary target is `delta_y`.",
            f"- The 'identifiable' subset (a_prev≠a_int) is vacuous here "
            f"(count={s['identifiable_count']}/{s['n_eval_tuples']}, a_prev always NOOP) "
            "— it cannot separate distinguishable executed actions, so P2 breaks down by "
            "stuck-vs-free (physical branch difference) instead.",
        ]
        (self.out / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")

    # -------------------------------------------------- Step 2 / A0, A1
    def step_a0(self):
        m = TR.tiny_overfit(self.train_frames, self.train_py, self.out / "A0",
                            device=self.device, steps=800)
        self.gate("A0 Tiny overfit", "PASS" if m["pass"] else "FAIL",
                  "A0/tiny_overfit_metrics.json, A0/tiny_overfit_gallery.png",
                  "" if m["pass"] else "fix AE recipe before normal run",
                  f"loss {m['loss_start']:.3f}->{m['loss_end']:.3f} "
                  f"(x{m['loss_drop_ratio']:.0f}), recon/target var "
                  f"{m['recon_to_target_var_ratio']:.2f}")
        return m["pass"]

    def step_a1(self):
        cfg = AEConfig(steps=self.steps, lat_ch=16)
        self.ae, hist, ckpt = TR.train(self.train_frames, self.train_seeds,
                                       self.train_py, self.out / "A1", cfg,
                                       device=self.device)
        # judge A1: the meaningful check is that the held-out recon RECOVERS the paddle
        # (not a constant-background collapse). MSE val plateaus within a few hundred
        # steps, so require only that it did not INCREASE, plus paddle recovery.
        val0, val1 = hist["val"][0], hist["val"][-1]
        with_paddle = self._heldout_paddle_recovered()
        ok = (val1 <= 1.05 * val0) and with_paddle["paddle_recovered"]
        self.gate("A1 Small AE training", "PASS" if ok else "FAIL",
                  "A1/ae_checkpoint.pt, A1/reconstruction_curves.png, "
                  "A1/heldout_reconstruction_gallery.png, A1/paddle_crop_gallery.png",
                  "" if ok else "reconstruction inadequate",
                  f"val L1 {val0:.4f}->{val1:.4f}; paddle-region error explained "
                  f"{with_paddle['paddle_var_explained']:.2f}")
        self.a1_paddle = with_paddle
        return ok

    def _heldout_paddle_recovered(self):
        """On eval frame_t, does the recon capture the paddle (right band) or collapse
        to background? Measure fraction of paddle-band pixel VARIANCE explained."""
        import torch
        col = TR.player_paddle_column(self.b.cur_frame)
        F = self.b.cur_frame
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(F)).to(self.device)
            recon, _ = self.ae(x)
        recon = ((recon.squeeze(1).cpu().numpy() + 1) / 2 * 255)
        band_o = F[:, :, col].astype(np.float32)
        band_r = recon[:, :, col]
        # variance across samples per pixel: how much of the true spatial-temporal
        # variation the recon reproduces (1 - residual_var/true_var)
        true_var = band_o.var(0).mean()
        resid_var = (band_o - band_r).var(0).mean()
        ve = float(1.0 - resid_var / (true_var + 1e-9))
        return {"paddle_var_explained": ve, "paddle_recovered": bool(ve > 0.15)}

    # -------------------------------------------------- build reps once
    def build_reps(self, eye):
        b = self.b
        self.reps = Reps(b, self.ae, eye, self.device)
        # cache per-frame-set encodings
        self.enc = {
            "cur": {
                "object_state": self.reps.object_state_1f(),
                "jepa_tokens": self.reps.jepa_tokens_1f(b.cur_frame),
                "jepa_prequant": self.reps.jepa_prequant_1f(b.cur_frame),
                "ae_recon": self.reps.ae_1f(b.cur_frame),
            },
            "fac": {
                "object_state": b.o_fac.astype(np.float64),
                "jepa_tokens": self.reps.jepa_tokens_1f(b.fac_frame),
                "jepa_prequant": self.reps.jepa_prequant_1f(b.fac_frame),
                "ae_recon": self.reps.ae_1f(b.fac_frame),
            },
        }

    # -------------------------------------------------- Step 3 / P1
    def step_p1(self):
        b, tr, te = self.b, self.tr, self.te
        y = b.py_k
        rows = []
        for rep in REPRESENTATIONS:
            X = self.enc["cur"][rep]
            for mdl in ("ridge", "mlp"):
                r = PB.reg_probe(X, y, tr, te, model_name=mdl, seed=0)
                rows.append({"rep": rep, "model": mdl, **{k: r[k] for k in
                            ("test_mae", "test_nmae", "test_r2", "train_mae",
                             "baseline_mae", "shuffled_mae")},
                            "ci_lo": r["test_mae_ci"][0], "ci_hi": r["test_mae_ci"][1]})
                if mdl == "ridge":
                    self.compare.setdefault(rep, {})["position_mae"] = r["test_mae"]
                    if rep == "ae_recon":
                        self._p1_scatter(r), self._p1_bins(r)
        self._wcsv("position_probe_metrics_rows.csv", rows,
                   ["rep", "model", "test_mae", "test_nmae", "test_r2", "train_mae",
                    "baseline_mae", "shuffled_mae", "ci_lo", "ci_hi"])
        CM.write_json(self.out / "position_probe_metrics.json", {"rows": rows})
        ae = [r for r in rows if r["rep"] == "ae_recon" and r["model"] == "ridge"][0]
        jt = [r for r in rows if r["rep"] == "jepa_tokens" and r["model"] == "ridge"][0]
        beats_shuffle = ae["test_mae"] < 0.8 * ae["shuffled_mae"]
        ok = beats_shuffle
        self.gate("P1 Position", "PASS" if ok else "FAIL",
                  "position_probe_metrics.json, position_probe_scatter.png, "
                  "position_error_by_bin.png",
                  "" if ok else "AE latent lacks paddle position",
                  f"AE MAE {ae['test_mae']:.2f}px (shuffled {ae['shuffled_mae']:.2f}); "
                  f"JEPA-tokens {jt['test_mae']:.2f}px; object-state "
                  f"{self.compare['object_state']['position_mae']:.2f}px")
        return ok

    def _p1_scatter(self, r):
        plt = _plt()
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(r["y_te"], r["pred_te"], s=14, alpha=0.6)
        lim = [min(r["y_te"].min(), r["pred_te"].min()),
               max(r["y_te"].max(), r["pred_te"].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("true player_y (screen px)")
        ax.set_ylabel("predicted (AE latent, ridge)")
        ax.set_title(f"P1 AE position probe  MAE={r['test_mae']:.2f}px")
        fig.tight_layout(); fig.savefig(self.out / "position_probe_scatter.png", dpi=110)
        plt.close(fig)

    def _p1_bins(self, r):
        plt = _plt()
        rows = PB.error_by_bin(r["pred_te"], r["y_te"], n_bins=8)
        CM.write_json(self.out / "position_error_by_bin.json", {"bins": rows})
        centers = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(centers, [b["mae"] for b in rows],
               width=(centers[1] - centers[0]) * 0.8)
        ax.set_xlabel("player_y bin (screen px)"); ax.set_ylabel("MAE (px)")
        ax.set_title("P1 AE position error by paddle-position bin")
        fig.tight_layout(); fig.savefig(self.out / "position_error_by_bin.png", dpi=110)
        plt.close(fig)

    # -------------------------------------------------- Step 4 / P2
    def step_p2(self):
        b, tr, te = self.b, self.tr, self.te
        dy = b.delta_y
        dir_y = np.sign(dy).astype(int)
        # boundary: paddle within one paddle-step of the observed extremes at t OR t+1
        step_px = 4.586
        lo, hi = b.py_k.min(), b.py_k.max()
        boundary = ((b.py_k <= lo + step_px) | (b.py_k >= hi - step_px)
                    | (b.py_k1 <= lo + step_px) | (b.py_k1 >= hi - step_px))
        rng = np.random.default_rng(0)
        rows = []
        for rep in REPRESENTATIONS:
            Xt, Xt1 = self.enc["cur"][rep], self.enc["fac"][rep]
            X2 = np.concatenate([Xt, Xt1], 1)
            # regression of delta_y
            rr = PB.reg_probe(X2, dy, tr, te, model_name="ridge", seed=0)
            # direction classification
            rc = PB.clf_probe(X2, dir_y, tr, te, seed=0)
            # --- controls ---
            # (a) mismatched frame_t1: permute the t1 block within the split
            Xt1_perm = Xt1.copy()
            perm_tr = rng.permutation(np.where(tr)[0])
            perm_te = rng.permutation(np.where(te)[0])
            Xt1_perm[np.where(tr)[0]] = Xt1[perm_tr]
            Xt1_perm[np.where(te)[0]] = Xt1[perm_te]
            X2m = np.concatenate([Xt, Xt1_perm], 1)
            rr_mis = PB.reg_probe(X2m, dy, tr, te, model_name="ridge", seed=0)
            # (b) static-only: frame_t alone (does delta need t1 at all?)
            rr_static = PB.reg_probe(Xt, dy, tr, te, model_name="ridge", seed=0)
            rows.append({
                "rep": rep,
                "dy_r2": rr["test_r2"], "dy_mae": rr["test_mae"],
                "dy_shuffled_mae": rr["shuffled_mae"], "dy_baseline_mae": rr["baseline_mae"],
                "dy_mismatch_r2": rr_mis["test_r2"], "dy_static_r2": rr_static["test_r2"],
                "dir_acc": rc["test_acc"], "dir_baseline": rc["baseline_acc"],
                "dir_shuffled": rc["shuffled_acc"], "dir_ci_lo": rc["test_acc_ci"][0],
                "dir_ci_hi": rc["test_acc_ci"][1],
            })
            self.compare.setdefault(rep, {})["dy_r2"] = rr["test_r2"]
            self.compare[rep]["dir_acc"] = rc["test_acc"]
            if rep == "ae_recon":
                self._p2_scatter(rr, rc)
        # subset breakdown (AE + JEPA tokens) on all / non-boundary / free / stuck
        subrows = self._p2_subsets(boundary)
        self._wcsv("displacement_probe_rows.csv", rows,
                   list(rows[0].keys()))
        self._wcsv("displacement_subset_breakdown.csv", subrows,
                   list(subrows[0].keys()))
        CM.write_json(self.out / "displacement_probe_metrics.json",
                      {"rows": rows, "subsets": subrows,
                       "n_boundary": int(boundary.sum()),
                       "note": "identifiable/free-action subset is vacuous (a_prev always "
                               "NOOP); using stuck-vs-free physical-branch split instead."})
        ae = [r for r in rows if r["rep"] == "ae_recon"][0]
        jt = [r for r in rows if r["rep"] == "jepa_tokens"][0]
        # PASS: AE has delta_y info (dir beats baseline+shuffle) AND drops under mismatch
        has_info = (ae["dir_acc"] > ae["dir_baseline"] + 0.03
                    and ae["dir_acc"] > ae["dir_shuffled"] + 0.03)
        drops = ae["dy_mismatch_r2"] < ae["dy_r2"] - 0.02
        ok = has_info and drops
        self.gate("P2 Displacement", "PASS" if ok else "FAIL",
                  "displacement_probe_metrics.json, displacement_scatter.png, "
                  "displacement_subset_breakdown.csv",
                  "" if ok else "AE latent lacks realized displacement info",
                  f"AE dir acc {ae['dir_acc']:.2f} (base {ae['dir_baseline']:.2f}), "
                  f"dy R2 {ae['dy_r2']:.2f}->{ae['dy_mismatch_r2']:.2f} mismatch; "
                  f"JEPA-tok dir {jt['dir_acc']:.2f} R2 {jt['dy_r2']:.2f}")
        return ok

    def _p2_subsets(self, boundary):
        b, tr, te = self.b, self.tr, self.te
        dy = b.delta_y; dir_y = np.sign(dy).astype(int)
        free = b.stuck == 0
        subsets = {"all": np.ones_like(free, bool), "non_boundary": ~boundary,
                   "free": free, "stuck": b.stuck == 1}
        out = []
        for rep in ("ae_recon", "jepa_tokens", "object_state"):
            Xt, Xt1 = self.enc["cur"][rep], self.enc["fac"][rep]
            X2 = np.concatenate([Xt, Xt1], 1)
            for name, mask in subsets.items():
                trm, tem = tr & mask, te & mask
                if tem.sum() < 6 or len(np.unique(dir_y[trm])) < 2:
                    out.append({"rep": rep, "subset": name, "n_test": int(tem.sum()),
                                "dir_acc": "", "dir_baseline": "", "dy_r2": ""})
                    continue
                rc = PB.clf_probe(X2, dir_y, trm, tem, seed=0)
                rr = PB.reg_probe(X2, dy, trm, tem, model_name="ridge", seed=0)
                out.append({"rep": rep, "subset": name, "n_test": int(tem.sum()),
                            "dir_acc": round(rc["test_acc"], 3),
                            "dir_baseline": round(rc["baseline_acc"], 3),
                            "dy_r2": round(rr["test_r2"], 3)})
        return out

    def _p2_scatter(self, rr, rc):
        plt = _plt()
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].scatter(rr["y_te"], rr["pred_te"], s=14, alpha=0.6)
        lim = [rr["y_te"].min(), rr["y_te"].max()]
        ax[0].plot(lim, lim, "k--", lw=1)
        ax[0].set_xlabel("true Δplayer_y (px)"); ax[0].set_ylabel("predicted (AE)")
        ax[0].set_title(f"P2 AE Δy regression R²={rr['test_r2']:.2f}")
        # direction confusion
        from itertools import product
        labs = [-1, 0, 1]
        cm = np.zeros((3, 3), int)
        for t, p in zip(rc["y_te"], rc["pred_te"]):
            cm[labs.index(t), labs.index(p)] += 1
        im = ax[1].imshow(cm, cmap="Blues")
        ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["down", "stay", "up"])
        ax[1].set_yticks(range(3)); ax[1].set_yticklabels(["down", "stay", "up"])
        for i, j in product(range(3), range(3)):
            ax[1].text(j, i, cm[i, j], ha="center", va="center")
        ax[1].set_title(f"P2 AE direction acc={rc['test_acc']:.2f}")
        ax[1].set_xlabel("pred"); ax[1].set_ylabel("true")
        fig.tight_layout(); fig.savefig(self.out / "displacement_scatter.png", dpi=110)
        plt.close(fig)

    # -------------------------------------------------- Step 5 / P3
    def step_p3(self):
        b = self.b
        import torch
        valid = b.phys["valid_fac_cf"]
        different = valid & ~b.phys["identical_fac_cf"]   # 122 free: paddle outcome differs
        identical = valid & b.phys["identical_fac_cf"]    # 128 stuck: fac==cf
        metrics = {}
        rep_diffs = {}
        for rep in ("ae_recon", "jepa_prequant", "object_state"):
            if rep == "object_state":
                fac = b.o_fac; cf = b.o_cf
            else:
                enc = self.reps.ae_1f if rep == "ae_recon" else self.reps.jepa_prequant_1f
                fac = enc(b.fac_frame); cf = enc(b.cf_frame)
            d = np.linalg.norm(fac - cf, axis=1)
            rep_diffs[rep] = d
            md, mi = float(d[different].mean()), float(d[identical].mean())
            # separability: AUC of "different vs identical" using latent distance
            from sklearn.metrics import roc_auc_score
            yb = np.concatenate([np.ones(different.sum()), np.zeros(identical.sum())])
            sb = np.concatenate([d[different], d[identical]])
            auc = float(roc_auc_score(yb, sb)) if len(np.unique(yb)) > 1 else float("nan")
            metrics[rep] = {"mean_dist_different": md, "mean_dist_identical": mi,
                            "ratio": md / (mi + 1e-9), "auc_diff_vs_identical": auc}
            self.compare.setdefault(rep, {})["branch_auc"] = auc
        # spatial locality of the AE latent difference (which latent cells move?)
        loc = self._p3_spatial_locality(different)
        metrics["ae_spatial_locality"] = loc
        CM.write_json(self.out / "branch_metrics.json", metrics)
        self._p3_galleries(different, rep_diffs["ae_recon"])
        ae = metrics["ae_recon"]
        ok = ae["auc_diff_vs_identical"] > 0.6 and ae["ratio"] > 1.1
        self.gate("P3 Branch sensitivity", "PASS" if ok else "FAIL",
                  "branch_metrics.json, branch_pair_gallery.png, "
                  "latent_difference_gallery.png",
                  "" if ok else "AE cannot separate branch outcomes",
                  f"AE dist different/identical {ae['mean_dist_different']:.2f}/"
                  f"{ae['mean_dist_identical']:.2f} (AUC {ae['auc_diff_vs_identical']:.2f}); "
                  f"JEPA-prequant AUC {metrics['jepa_prequant']['auc_diff_vs_identical']:.2f}")
        return ok

    def _p3_spatial_locality(self, different):
        """For different-outcome pairs, per-latent-cell mean |Δ| — is it near the paddle
        band (right columns) rather than uniform/ball-dominated?"""
        import torch
        b = self.b
        with torch.no_grad():
            zf = self.ae.encode(torch.from_numpy(
                np.ascontiguousarray(b.fac_frame[different])).to(self.device))
            zc = self.ae.encode(torch.from_numpy(
                np.ascontiguousarray(b.cf_frame[different])).to(self.device))
        diff = (zf - zc).abs().mean(1).cpu().numpy()   # (n, 6, 6) over channels
        cell = diff.mean(0)                            # (6,6)
        # right band = last 2 columns of the 6x6 grid (paddle side)
        right = float(cell[:, -2:].mean()); left = float(cell[:, :-2].mean())
        return {"cell_grid": cell.tolist(),
                "right_band_mean": right, "rest_mean": left,
                "right_over_rest": right / (left + 1e-9)}

    def _p3_galleries(self, different, ae_dist):
        plt = _plt()
        b = self.b
        idx = np.where(different)[0]
        order = idx[np.argsort(-ae_dist[idx])][:6]   # top-6 by AE latent distance
        fig, ax = plt.subplots(3, len(order), figsize=(1.7 * len(order), 5.4))
        for j, i in enumerate(order):
            for r, (img, lab) in enumerate([
                    (b.fac_frame[i], "factual"), (b.cf_frame[i], "counterfactual"),
                    (np.abs(b.fac_frame[i].astype(float) - b.cf_frame[i]), "|Δpixels|")]):
                a = ax[r, j]
                a.imshow(img, cmap="gray" if r < 2 else "magma")
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.set_title(f"Δpy={b.phys['dplayer_y_fac_cf'][i]:.0f}", fontsize=8)
                if j == 0:
                    a.set_ylabel(lab, fontsize=9)
        fig.suptitle("P3 branch pairs (different paddle outcome), top-6 by AE latent dist")
        fig.tight_layout(); fig.savefig(self.out / "branch_pair_gallery.png", dpi=110)
        plt.close(fig)
        # latent difference heatmap gallery (AE spatial |Δ| upsampled)
        import torch
        with torch.no_grad():
            zf = self.ae.encode(torch.from_numpy(
                np.ascontiguousarray(b.fac_frame[order])).to(self.device))
            zc = self.ae.encode(torch.from_numpy(
                np.ascontiguousarray(b.cf_frame[order])).to(self.device))
        dd = (zf - zc).abs().mean(1).cpu().numpy()   # (k,6,6)
        fig, ax = plt.subplots(1, len(order), figsize=(1.7 * len(order), 2.2))
        for j in range(len(order)):
            ax[j].imshow(dd[j], cmap="magma")
            ax[j].set_xticks([]); ax[j].set_yticks([])
        fig.suptitle("P3 AE latent |Δ| per 6×6 cell (paddle = right cols)")
        fig.tight_layout(); fig.savefig(self.out / "latent_difference_gallery.png", dpi=110)
        plt.close(fig)

    # -------------------------------------------------- comparison + gate table
    def finish(self):
        rep_label = {"object_state": "Object state (oracle)",
                     "jepa_tokens": "Discrete-JEPA (tokens)",
                     "jepa_prequant": "Discrete-JEPA (pre-quant)",
                     "ae_recon": "Reconstruction AE"}
        lines = ["# Representation comparison (Rung 3 diagnostic)\n",
                 "| Representation | Position MAE (px)↓ | Δy R²↑ | Δy dir acc↑ | "
                 "Branch AUC↑ |",
                 "|---|---:|---:|---:|---:|"]
        for rep in ("object_state", "jepa_tokens", "jepa_prequant", "ae_recon"):
            c = self.compare.get(rep, {})
            lines.append(
                f"| {rep_label[rep]} | {_f(c.get('position_mae'))} | "
                f"{_f(c.get('dy_r2'))} | {_f(c.get('dir_acc'))} | "
                f"{_f(c.get('branch_auc'))} |")
        (self.out / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")
        # gate table
        g = ["# Gate table\n", "| Gate | Result | Evidence | Blocking |",
             "|---|---|---|---|"]
        for r in self.gate_rows:
            g.append(f"| {r['gate']} | {r['result']} | {r['evidence']} | "
                     f"{r['blocking'] or '—'} |")
        (self.out / "gate_table.md").write_text("\n".join(g), encoding="utf-8")
        CM.write_json(self.out / "gate_table.json", {"gates": self.gate_rows,
                                                     "comparison": self.compare})
        sprint("\n".join(lines)); sprint("\n".join(g))

    def _wcsv(self, name, rows, fields):
        with open(self.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})


def _f(x):
    return "—" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pong_counterfactual/cjepa_rung3_diag_ae/outputs/main")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lat-spatial", type=int, default=6)
    ap.add_argument("--eye", default=str(CM.DEFAULT_EYE))
    args = ap.parse_args()
    d = Diagnostic(Path(args.out), args.steps, args.device, args.lat_spatial)
    if not d.step_d0():
        print("D0 FAILED — stop."); return
    if not d.step_a0():
        print("A0 FAILED — stop before normal run."); return
    d.step_a1()
    eye, _ = load_eye_from_path(Path(args.eye), args.device)
    d.build_reps(eye)
    d.step_p1()
    d.step_p2()
    d.step_p3()
    d.finish()


if __name__ == "__main__":
    main()
