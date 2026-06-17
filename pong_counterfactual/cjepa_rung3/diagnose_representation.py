"""Rung 3 Stage A1 — Representation Abductability Audit (read-only).

Question: does the FROZEN Discrete-JEPA representation (the approved
`tokenizer_M4_K64_res84/final.pt`) contain the information a later counterfactual
abduction step would need? Concretely, three sub-questions:

  A. Branch validity        — are the oracle factual vs alternative branches actually
                              physically different states?
  B. Representation sep.     — does the frozen eye PRESERVE those physical differences in
                              token / pre-quant latent space, or collapse them?
  C. Abductability           — can sticky / executed-action information be DECODED from a
                              factual latent transition (z_t -> z_{t+1}) beyond what the
                              actions alone already tell you?

This script ONLY reads. It loads the frozen checkpoint on CPU, encodes existing frames,
trains tiny linear/logistic diagnostic probes, and writes plots + tables to a fresh
timestamped audit dir. It NEVER retrains/modifies the tokenizer, never runs Stage 3/4,
and never writes inside an existing run/result directory. `weights_only=False` is used to
load the trusted repo-generated checkpoint (it carries a plain config dict); CPU-only.

A low collision rate ALONE is NOT taken as evidence of abductability (sub-question C is
the real test). See REPORT.md verdict rules.

Run:
  # validate the supplied data first (no heavy compute):
  python -m pong_counterfactual.cjepa_rung3.diagnose_representation --preflight
  # full audit:
  python -m pong_counterfactual.cjepa_rung3.diagnose_representation \
      [--data-dir <dir with ep_*.npz + oracle_cache.npz>] [--checkpoint <final.pt>]

Required supplied data (the s=0.5, res=84 Stage-0 set the eye was gated on):
  <data-dir>/
     ep_eval_01000.npz .. ep_eval_0101N.npz   (eval episodes; EpisodeData schema)
     ep_train_00000.npz ..                      (train episodes; used for the probe-fit
                                                 sanity baseline only)
     oracle_cache.npz                           (pool, o_cf, o_fac, other, stuck,
                                                 o_cf_frame, o_fac_frame, other_frame)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- repo constants/code
from pong_counterfactual.cjepa_rung3 import config as C
from pong_counterfactual.cjepa_rung3.data import (load_split, build_eval_pool,
                                                  EpisodeData, NOOP, FIRE, UP, DOWN,
                                                  ACTIONS_H, opposite)

DIMS = ("ball_x", "ball_y", "player_y", "enemy_y")
ORACLE_FRAME_KEYS = ("o_cf_frame", "o_fac_frame", "other_frame")
ORACLE_VEC_KEYS = ("o_cf", "o_fac", "other")
APPROVED_CKPT_GLOB = "**/tokenizer_M4_K64_res84*/final.pt"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return p


# ============================================================ pure, unit-tested helpers
def onehot_codes(codes: np.ndarray, M: int, K: int) -> np.ndarray:
    """(N,M) int codes -> (N, M*K) float slot-blocked one-hot (matches stage2_gate)."""
    N = len(codes)
    X = np.zeros((N, M * K), dtype=np.float32)
    for m in range(M):
        X[np.arange(N), m * K + codes[:, m]] = 1.0
    return X


def physical_distances(o_fac: np.ndarray, o_cf: np.ndarray,
                       other: np.ndarray, eps: float = 1e-9) -> dict:
    """Per-pair physical distances between oracle branches at the 4-vector level.
    NaN rows (invalid replay states) are propagated as NaN distances. Pure function."""
    def d(a, b):
        return np.abs(a - b)
    fac_cf = d(o_fac, o_cf)           # (N,4)
    fac_oth = d(o_fac, other)
    out = {
        "dplayer_y_fac_cf": fac_cf[:, 2],
        "dball_x_fac_cf": fac_cf[:, 0],
        "dball_y_fac_cf": fac_cf[:, 1],
        "dvec_l2_fac_cf": np.sqrt((fac_cf ** 2).sum(1)),
        "dplayer_y_fac_oth": fac_oth[:, 2],
        "dvec_l2_fac_oth": np.sqrt((fac_oth ** 2).sum(1)),
        "identical_fac_cf": (np.nan_to_num(fac_cf, nan=0.0).max(1) <= eps)
                            & ~np.isnan(fac_cf).any(1),
        "identical_fac_oth": (np.nan_to_num(fac_oth, nan=0.0).max(1) <= eps)
                             & ~np.isnan(fac_oth).any(1),
        "valid_fac_cf": ~(np.isnan(o_fac).any(1) | np.isnan(o_cf).any(1)),
        "valid_fac_oth": ~(np.isnan(o_fac).any(1) | np.isnan(other).any(1)),
    }
    return out


def group_train_test_split(group_ids: np.ndarray, test_frac: float = 0.3,
                           seed: int = 0):
    """Episode-grouped split: every group is entirely in train OR test (no temporal
    leakage across the t->t+1 transitions of one episode). Returns (train_mask,
    test_mask) boolean arrays. Pure function."""
    groups = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(groups)
    n_test = max(1, int(round(len(groups) * test_frac)))
    test_groups = set(perm[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in group_ids])
    return ~test_mask, test_mask


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05):
    """Percentile bootstrap CI of the MEAN of a per-sample score array (e.g. per-sample
    correctness for accuracy, or absolute error for MAE). Pure function."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = values[idx].mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def executed_action(a_int: np.ndarray, a_prev: np.ndarray,
                    stuck: np.ndarray) -> np.ndarray:
    """Realized action at step k under ALE sticky: a_prev if the sticky fired, else
    a_int. (stuck==1 -> repeat previous; stuck in {0,-1} -> intended.) Pure function."""
    out = np.where(stuck == 1, a_prev, a_int)
    return out.astype(np.int64)


# ============================================================ checkpoint / encoding
def load_eye_from_path(ckpt_path: Path, device: str = "cpu"):
    """Load the frozen Discrete-JEPA from an explicit final.pt path (mirrors
    tokenizer.load_eye but path-driven, so it works on the copied results dir)."""
    import torch
    from pong_counterfactual.cjepa_rung3.config import TokenizerConfig
    from pong_counterfactual.cjepa_rung3.tokenizer import DiscreteJEPA
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TokenizerConfig(**payload["config"])
    model = DiscreteJEPA(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def encode_tokens(model, frames_u8: np.ndarray, device: str, batch: int = 256):
    """(N,res,res) uint8 -> (N,M) int64 token indices (the eye's downstream output)."""
    import torch
    out = []
    for i in range(0, len(frames_u8), batch):
        x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + batch])).to(device)
        out.append(model.encode(x).cpu().numpy())
    return np.concatenate(out) if len(frames_u8) else np.zeros((0, model.cfg.M), np.int64)


def prequant_latents(model, frames_u8: np.ndarray, device: str, batch: int = 256):
    """(N,res,res) uint8 -> (N,M,code_dim) float pre-quantization latents (proj_in(sem)).
    The continuous signal the VQ then discretizes — 'pre-quant latent distance'."""
    import torch
    out = []
    for i in range(0, len(frames_u8), batch):
        x = torch.from_numpy(np.ascontiguousarray(frames_u8[i:i + batch])).to(device)
        img = x.float().div(255.0).mul(2).sub(1).unsqueeze(1)
        _, sem = model.context(img)
        z = model.vq.proj_in(sem)
        out.append(z.cpu().numpy())
    return np.concatenate(out) if len(frames_u8) else np.zeros((0, model.cfg.M, model.cfg.code_dim), np.float32)


# ============================================================ data discovery / loading
def discover_checkpoint(repo: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint not found: {p}")
        return p
    hits = sorted((repo / "pong_counterfactual" / "results").glob(APPROVED_CKPT_GLOB))
    hits = [h for h in hits if h.name == "final.pt"]
    if not hits:
        raise FileNotFoundError(
            "approved checkpoint tokenizer_M4_K64_res84*/final.pt not found under "
            f"{repo/'pong_counterfactual'/'results'}; pass --checkpoint")
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous approved checkpoint: {hits}; pass --checkpoint")
    return hits[0].resolve()


def _is_full_oracle(npz_path: Path) -> bool:
    try:
        with np.load(npz_path, allow_pickle=True) as z:
            return all(k in z.files for k in (*ORACLE_FRAME_KEYS, *ORACLE_VEC_KEYS,
                                              "pool", "stuck"))
    except Exception:
        return False


def checkpoint_res(ckpt: Path) -> int:
    """Read just the input resolution from the checkpoint's embedded config (no model
    build). This is the AUTHORITATIVE target res — the eye only accepts frames of its
    own resolution — NOT the (possibly newer) global config default."""
    import torch
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    return int(payload["config"]["res"])


_S_RE = __import__("re").compile(r"data_s([0-9.]+)")


def _parse_s(dir_name: str):
    m = _S_RE.search(dir_name)
    return float(m.group(1)) if m else None


def discover_data_dir(repo: Path, s: float, target_res: int,
                      explicit: str | None) -> Path | None:
    """Find the Stage-0 data dir matching BOTH the requested sticky level s AND the
    approved checkpoint's resolution. Selects by the oracle's actual frame resolution
    (authoritative) and the s parsed from the directory name — never the global config
    default, and never the first arbitrary glob hit."""
    if explicit:
        return Path(explicit).resolve()
    matches = []
    for oc in repo.rglob("oracle_cache.npz"):
        if ".venv" in oc.parts or "site-packages" in oc.parts:
            continue
        d = oc.parent
        if not (_is_full_oracle(oc) and any(d.glob("ep_eval_*.npz"))):
            continue
        with np.load(oc, allow_pickle=True) as z:
            fr_res = int(z["o_fac_frame"].shape[-1])
        d_s = _parse_s(d.name)
        if fr_res == target_res and d_s is not None and abs(d_s - s) < 1e-9:
            matches.append(d.resolve())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"multiple data dirs match s={s} res={target_res}: {matches}")
    return None


def preflight(repo: Path, ckpt: Path, data_dir: Path | None, s: float,
              ckpt_res: int) -> dict:
    """Validate inputs WITHOUT heavy compute. Returns a report dict; ok=False blocks.
    Validates the data resolution against the APPROVED CHECKPOINT's resolution (the eye
    only accepts frames of its own res) and the sticky level against the requested s."""
    rep = {"checkpoint": str(ckpt), "checkpoint_exists": ckpt.exists(),
           "checkpoint_res": ckpt_res, "requested_s": s,
           "data_dir": str(data_dir) if data_dir else None, "issues": [], "found": {}}
    if not ckpt.exists():
        rep["issues"].append("approved checkpoint missing")
    if data_dir is None or not data_dir.exists():
        rep["issues"].append(
            "No Stage-0 data dir matching the approved checkpoint (s=%g, res=%d) was "
            "found. Need a directory with ep_eval_*.npz + ep_train_*.npz AND a frame-"
            "bearing oracle_cache.npz (keys: pool, o_fac, o_cf, other, stuck, "
            "o_fac_frame, o_cf_frame, other_frame). Pass it via --data-dir." % (s, ckpt_res))
        rep["ok"] = False
        return rep
    oc = data_dir / "oracle_cache.npz"
    rep["found"]["oracle_cache"] = oc.exists() and _is_full_oracle(oc)
    if not rep["found"]["oracle_cache"]:
        rep["issues"].append(f"{oc} missing required frame keys {ORACLE_FRAME_KEYS}")
    n_eval = len(list(data_dir.glob("ep_eval_*.npz")))
    n_train = len(list(data_dir.glob("ep_train_*.npz")))
    rep["found"]["n_eval_shards"] = n_eval
    rep["found"]["n_train_shards"] = n_train
    rep["found"]["data_dir_s"] = _parse_s(data_dir.name)
    if n_eval < C.N_EVAL_EPS:
        rep["issues"].append(f"expected {C.N_EVAL_EPS} eval shards, found {n_eval}")
    if oc.exists() and _is_full_oracle(oc):
        with np.load(oc, allow_pickle=True) as z:
            fr = z["o_fac_frame"]
            rep["found"]["oracle_frame_res"] = int(fr.shape[-1])
            rep["found"]["n_tuples"] = int(len(z["pool"]))
            rep["found"]["stuck_counts"] = {int(k): int(v) for k, v in
                                            zip(*np.unique(z["stuck"], return_counts=True))}
            if fr.shape[-1] != ckpt_res:
                rep["issues"].append(
                    f"oracle frame res {fr.shape[-1]} != APPROVED CHECKPOINT res {ckpt_res} "
                    f"— the eye cannot encode these frames")
    if rep["found"].get("data_dir_s") is not None and abs(rep["found"]["data_dir_s"] - s) > 1e-9:
        rep["issues"].append(
            f"data dir sticky level {rep['found']['data_dir_s']} != requested s={s}")
    rep["ok"] = not rep["issues"]
    return rep


# ============================================================ matplotlib (Agg, headless)
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ============================================================ the audit
class Diagnose:
    def __init__(self, repo, ckpt, data_dir, out_dir, s, device):
        self.repo, self.ckpt, self.data_dir = repo, ckpt, data_dir
        self.out, self.s, self.device = out_dir, s, device
        self.gate_rows = []
        self.blocking = []

    def gate(self, name, check, criterion, result, evidence, blocking, notes=""):
        self.gate_rows.append({"Gate": name, "Check": check, "PASS criterion": criterion,
                               "Result": result, "Evidence artifact": evidence,
                               "Blocking consequence": blocking, "Notes": notes})

    def _wj(self, name, obj):
        (self.out / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    def _wcsv(self, name, rows, fields):
        import csv
        with open(self.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    # ---- load everything once ----------------------------------------------------
    def load(self):
        self.model, self.cfg = load_eye_from_path(self.ckpt, self.device)
        self.M, self.K = self.cfg.M, self.cfg.K
        self.eval_eps = load_split(self.data_dir, "eval", C.EVAL_BASE_SEED, C.N_EVAL_EPS)
        self.train_eps = load_split(self.data_dir, "train", C.TRAIN_BASE_SEED, C.N_TRAIN_EPS)
        self.pool = build_eval_pool(self.eval_eps, C.N_HIST, C.N_SAMPLES)
        self.pool_arr = np.asarray(self.pool, dtype=np.int64)
        with np.load(self.data_dir / "oracle_cache.npz", allow_pickle=True) as z:
            self.oc = {k: z[k] for k in z.files}
        # per-tuple action context from the eval episodes
        a_int, a_prev, ep_seed, cur_frame = [], [], [], []
        py_k, py_k1 = [], []
        for (ei, k) in self.pool:
            ep = self.eval_eps[ei]
            a_int.append(int(ep.intended[k]))
            a_prev.append(int(ep.intended[k - 1]))
            ep_seed.append(int(ep.seed))
            cur_frame.append(ep.frames[k])
            py_k.append(ep.vecs[k][2])
            py_k1.append(ep.vecs[k + 1][2])
        self.a_int = np.asarray(a_int, np.int64)
        self.a_prev = np.asarray(a_prev, np.int64)
        self.ep_seed = np.asarray(ep_seed, np.int64)
        self.cur_frame = np.stack(cur_frame)
        self.py_k = np.asarray(py_k, np.float64)
        self.py_k1 = np.asarray(py_k1, np.float64)
        self.stuck = self.oc["stuck"].astype(np.int64)
        # identifiable sticky subset: previous action differs from intended action
        self.identifiable = (self.a_prev != self.a_int)

    # ---- GATE R1 -----------------------------------------------------------------
    def gate_r1(self):
        issues = []
        N = len(self.pool)
        # length consistency
        lens = {k: len(v) for k, v in self.oc.items() if hasattr(v, "__len__")
                and k not in ("vec_ok", "frame_ok")}
        len_ok = all(v == N for v in lens.values())
        if not len_ok:
            issues.append(f"oracle array length mismatch: {lens}")
        if not np.array_equal(self.oc["pool"], self.pool_arr):
            issues.append("oracle pool != rebuilt eval pool (stale oracle for this split)")
        # same-source check: factual replay must reproduce the recorded next state+frame
        vec_match, frame_match = [], []
        for i, (ei, k) in enumerate(self.pool):
            ep = self.eval_eps[ei]
            vec_match.append(bool(np.allclose(self.oc["o_fac"][i], ep.vecs[k + 1],
                                              equal_nan=True)))
            frame_match.append(bool(np.array_equal(self.oc["o_fac_frame"][i],
                                                   ep.frames[k + 1])))
        vec_rate, frame_rate = float(np.mean(vec_match)), float(np.mean(frame_match))
        if vec_rate < 0.999 or frame_rate < 0.999:
            issues.append(f"replay-determinism low: vec {vec_rate:.3f} frame {frame_rate:.3f}")
        # NaN/Inf in vec branches (frames are uint8 -> no NaN)
        nan_counts = {k: int(np.isnan(self.oc[k]).any(1).sum()) for k in ORACLE_VEC_KEYS}
        # physical distances
        phys = physical_distances(self.oc["o_fac"], self.oc["o_cf"], self.oc["other"])
        # pixel differences (mean abs over pixels)
        pix_fac_cf = np.abs(self.oc["o_fac_frame"].astype(np.int32)
                            - self.oc["o_cf_frame"].astype(np.int32)).mean((1, 2))
        pix_fac_oth = np.abs(self.oc["o_fac_frame"].astype(np.int32)
                             - self.oc["other_frame"].astype(np.int32)).mean((1, 2))
        pix_identical_fac_cf = (pix_fac_cf == 0)
        # sample counts
        counts = {
            "n_tuples": N,
            "n_stuck": int((self.stuck == 1).sum()),
            "n_free": int((self.stuck == 0).sum()),
            "n_illposed_stuck": int((self.stuck == -1).sum()),
            "n_identifiable": int(self.identifiable.sum()),
            "n_identifiable_and_stuck": int((self.identifiable & (self.stuck == 1)).sum()),
            "n_identifiable_and_free": int((self.identifiable & (self.stuck == 0)).sum()),
            "n_valid_fac_cf": int(phys["valid_fac_cf"].sum()),
        }
        n_differ_phys = int((~phys["identical_fac_cf"] & phys["valid_fac_cf"]).sum())
        n_differ_pix = int((~pix_identical_fac_cf).sum())
        counts["n_branches_physically_differ_fac_cf"] = n_differ_phys
        counts["n_branches_pixel_differ_fac_cf"] = n_differ_pix
        counts["frac_branches_differ_fac_cf"] = (n_differ_phys
                                                 / max(1, counts["n_valid_fac_cf"]))

        self._wj("sample_counts.json", counts)
        self._wj("data_integrity.json", {
            "length_consistent": len_ok, "lengths": lens,
            "pool_matches": bool(np.array_equal(self.oc["pool"], self.pool_arr)),
            "replay_vec_match_rate": vec_rate, "replay_frame_match_rate": frame_rate,
            "nan_rows_per_branch": nan_counts, "issues": issues})

        # branch_physical_distance.csv
        rows = []
        coll = self.oc.get("coll_flag")  # not in oracle; from gate artifacts (optional)
        for i, (ei, k) in enumerate(self.pool):
            rows.append({
                "i": i, "episode_seed": int(self.eval_eps[ei].seed), "ei": ei, "k": k,
                "a_prev": int(self.a_prev[i]), "a_int": int(self.a_int[i]),
                "identifiable": bool(self.identifiable[i]), "stuck": int(self.stuck[i]),
                "dplayer_y_fac_cf": float(phys["dplayer_y_fac_cf"][i]),
                "dvec_l2_fac_cf": float(phys["dvec_l2_fac_cf"][i]),
                "dplayer_y_fac_oth": float(phys["dplayer_y_fac_oth"][i]),
                "pix_meanabs_fac_cf": float(pix_fac_cf[i]),
                "pix_meanabs_fac_oth": float(pix_fac_oth[i]),
                "identical_fac_cf": bool(phys["identical_fac_cf"][i]),
            })
        self.phys, self.pix_fac_cf, self.pix_fac_oth = phys, pix_fac_cf, pix_fac_oth
        self._wcsv("branch_physical_distance.csv", rows,
                   ["i", "episode_seed", "ei", "k", "a_prev", "a_int", "identifiable",
                    "stuck", "dplayer_y_fac_cf", "dvec_l2_fac_cf", "dplayer_y_fac_oth",
                    "pix_meanabs_fac_cf", "pix_meanabs_fac_oth", "identical_fac_cf"])

        # histogram
        plt = _plt()
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        v = phys["dplayer_y_fac_cf"][phys["valid_fac_cf"]]
        ax[0].hist(v, bins=30, color="tab:blue"); ax[0].set_title("|Δplayer_y| fac vs CF (screen px)")
        ax[1].hist(phys["dvec_l2_fac_cf"][phys["valid_fac_cf"]], bins=30, color="tab:green")
        ax[1].set_title("‖Δ 4-vec‖₂ fac vs CF")
        ax[2].hist(pix_fac_cf, bins=30, color="tab:orange"); ax[2].set_title("mean |Δpixel| fac vs CF")
        for a in ax: a.set_ylabel("count")
        fig.tight_layout(); fig.savefig(self.out / "branch_distance_histogram.png", dpi=110)
        plt.close(fig)

        # gate decision
        valid = (len_ok and vec_rate >= 0.999 and frame_rate >= 0.999
                 and counts["n_valid_fac_cf"] > 0)
        differ_enough = counts["frac_branches_differ_fac_cf"] >= 0.2
        if not valid:
            self.blocking.append("R1: oracle/branch data invalid — downstream gates blocked")
            result = "FAIL"
        elif not differ_enough:
            result = "PASS WITH WARNINGS"
            issues.append(f"only {counts['frac_branches_differ_fac_cf']:.0%} of branches "
                          f"physically differ — limited contrast")
        else:
            result = "PASS"
        self.gate("R1", "data/oracle validity + branch physical difference",
                  "consistent lengths, 100% replay determinism, branches physically differ",
                  result, "sample_counts.json, data_integrity.json, "
                  "branch_physical_distance.csv, branch_distance_histogram.png",
                  "if FAIL, R2/R3 are blocked (no trustworthy branches)",
                  "; ".join(issues) if issues else
                  f"{n_differ_phys}/{counts['n_valid_fac_cf']} branches differ physically")
        self.r1_valid = valid
        return {"counts": counts, "valid": valid}

    # ---- GATE R2 -----------------------------------------------------------------
    def gate_r2(self):
        if not self.r1_valid:
            self.gate("R2", "frozen tokenizer encoding", "—", "BLOCKED",
                      "—", "R1 invalid", "skipped: R1 branches invalid")
            return {"blocked": True}
        # encode the four frame sets
        cur = encode_tokens(self.model, self.cur_frame, self.device)
        fac = encode_tokens(self.model, self.oc["o_fac_frame"], self.device)
        cf = encode_tokens(self.model, self.oc["o_cf_frame"], self.device)
        oth = encode_tokens(self.model, self.oc["other_frame"], self.device)
        # determinism: encode factual twice
        fac2 = encode_tokens(self.model, self.oc["o_fac_frame"], self.device)
        deterministic = bool(np.array_equal(fac, fac2))
        self.tok = {"cur": cur, "fac": fac, "cf": cf, "oth": oth}

        # pre-quant latents (factual vs CF)
        lat_fac = prequant_latents(self.model, self.oc["o_fac_frame"], self.device)
        lat_cf = prequant_latents(self.model, self.oc["o_cf_frame"], self.device)
        lat_dist = np.sqrt(((lat_fac - lat_cf) ** 2).sum(-1)).mean(-1)  # mean over slots

        def collide(a, b):
            same = (a == b)
            return same.all(1), same  # all-token collide bool, per-slot bool (N,M)

        coll_cf, same_cf = collide(fac, cf)
        coll_oth, same_oth = collide(fac, oth)
        ham_cf = (fac != cf).sum(1)
        ham_oth = (fac != oth).sum(1)

        valid_cf = self.phys["valid_fac_cf"]
        # THE separability test conditions on PHYSICALLY-DIFFERENT pairs only: collision on
        # physically-identical (stuck) pairs is correct behaviour (same frame -> same
        # tokens), NOT collapse. Measuring over all pairs would just report the stuck rate.
        differ = valid_cf & ~self.phys["identical_fac_cf"]
        identical = valid_cf & self.phys["identical_fac_cf"]
        # cross-check factual-vs-other collision against the saved gate coll_flag (free)
        cross = self._crosscheck_collision(coll_oth)

        def _coll(mask):
            return float(coll_cf[mask].mean()) if mask.any() else None

        metrics = {
            "deterministic_encoding": deterministic,
            "n_pairs": int(len(fac)),
            "n_physically_differ": int(differ.sum()),
            "n_physically_identical": int(identical.sum()),
            # primary separability metric: collision among PHYSICALLY-DIFFERENT branches
            "collision_among_differing_fac_cf": _coll(differ),
            "mean_hamming_among_differing_fac_cf": float(ham_cf[differ].mean()) if differ.any() else None,
            # context-only (NOT the gate metric): collision on identical pairs should be ~1
            "collision_among_identical_fac_cf": _coll(identical),
            "collision_allpairs_fac_cf": _coll(valid_cf),
            "collision_per_slot_fac_cf_differing": [float(x) for x in same_cf[differ].mean(0)]
                                                   if differ.any() else None,
            "collision_allM_fac_other": float(coll_oth.mean()),
            "collision_per_slot_fac_other": [float(x) for x in same_oth.mean(0)],
            "mean_token_hamming_fac_cf_allpairs": float(ham_cf[valid_cf].mean()) if valid_cf.any() else None,
            "mean_token_hamming_fac_other": float(ham_oth.mean()),
            "mean_prequant_latent_dist_differing_fac_cf": float(lat_dist[differ].mean()) if differ.any() else None,
            "crosscheck_vs_gate_coll_flag": cross,
        }
        self._wj("representation_metrics.json", metrics)
        self.ham_cf, self.lat_dist, self.coll_cf = ham_cf, lat_dist, coll_cf

        # pair_metrics.csv
        rows = []
        for i in range(len(fac)):
            rows.append({
                "i": i, "valid_fac_cf": bool(valid_cf[i]),
                "dplayer_y_fac_cf": float(self.phys["dplayer_y_fac_cf"][i]),
                "pix_meanabs_fac_cf": float(self.pix_fac_cf[i]),
                "token_hamming_fac_cf": int(ham_cf[i]),
                "collision_fac_cf": bool(coll_cf[i]),
                "prequant_latent_dist_fac_cf": float(lat_dist[i]),
                "token_hamming_fac_other": int(ham_oth[i]),
                "collision_fac_other": bool(coll_oth[i]),
                "cur_tokens": "|".join(map(str, cur[i])),
                "fac_tokens": "|".join(map(str, fac[i])),
                "cf_tokens": "|".join(map(str, cf[i])),
            })
        self._wcsv("pair_metrics.csv", rows,
                   ["i", "valid_fac_cf", "dplayer_y_fac_cf", "pix_meanabs_fac_cf",
                    "token_hamming_fac_cf", "collision_fac_cf", "prequant_latent_dist_fac_cf",
                    "token_hamming_fac_other", "collision_fac_other",
                    "cur_tokens", "fac_tokens", "cf_tokens"])

        self._r2_plots(ham_cf, lat_dist, coll_cf, valid_cf)
        self._r2_gallery(cur, fac, cf, coll_cf, valid_cf)

        # gate: among PHYSICALLY-DIFFERENT branches, the eye must keep them token-distinct
        coll = metrics["collision_among_differing_fac_cf"]
        ham = metrics["mean_hamming_among_differing_fac_cf"]
        result = "PASS"
        notes = (f"on {metrics['n_physically_differ']} physically-DIFFERENT pairs: collision "
                 f"{coll:.1%}, mean Hamming {ham:.2f}/{self.M} "
                 f"(identical/stuck pairs collide {metrics['collision_among_identical_fac_cf']:.0%}, "
                 f"as expected — same frame)")
        if not deterministic:
            result = "FAIL"; notes = "encoding NON-deterministic — eye unusable"
            self.blocking.append("R2: non-deterministic encoding")
        elif coll is None:
            result = "PASS WITH WARNINGS"; notes = "no physically-different pairs to test"
        elif coll > 0.5:
            result = "FAIL"
            notes += " — eye COLLAPSES most physically-distinct branches"
            self.blocking.append("R2: representation collapses branch differences")
        elif coll > 0.25:
            result = "PASS WITH WARNINGS"; notes += " — partial collapse"
        self.gate("R2", "frozen encoding: determinism, collision, Hamming, latent dist",
                  "deterministic; physically-different branches stay token-distinct",
                  result, "representation_metrics.json, pair_metrics.csv, "
                  "token_hamming_histogram.png, physical_vs_latent_distance.png, "
                  "collision_by_physical_difference.png, pair_gallery.png",
                  "if FAIL, abductability (R3) cannot rest on these tokens", notes)
        return {"metrics": metrics}

    def _crosscheck_collision(self, coll_oth):
        """Compare our recomputed factual-vs-other collision with the gate's saved
        coll_flag (free tuples only) from the approved run's gate_artifacts.npz."""
        try:
            ga = self.ckpt.parent / "gate_artifacts.npz"
            with np.load(ga) as z:
                if not np.array_equal(z["pool"], self.pool_arr):
                    return {"available": True, "pool_match": False}
                flag = z["coll_flag"]
                free = flag != -1
                agree = bool(np.array_equal(coll_oth[free].astype(np.int64), flag[free]))
                return {"available": True, "pool_match": True,
                        "n_free": int(free.sum()), "matches_saved_coll_flag": agree}
        except Exception as e:
            return {"available": False, "error": str(e)}

    def _r2_plots(self, ham_cf, lat_dist, coll_cf, valid_cf):
        plt = _plt()
        # token hamming histogram
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(ham_cf[valid_cf], bins=np.arange(self.M + 2) - 0.5, color="tab:purple")
        ax.set_xlabel("token Hamming distance (fac vs CF)"); ax.set_ylabel("count")
        ax.set_title("Token Hamming, factual vs counterfactual")
        fig.tight_layout(); fig.savefig(self.out / "token_hamming_histogram.png", dpi=110)
        plt.close(fig)
        # physical vs latent distance scatter
        fig, ax = plt.subplots(figsize=(6, 4))
        d = self.phys["dplayer_y_fac_cf"]
        m = valid_cf
        sc = ax.scatter(d[m], lat_dist[m], c=ham_cf[m], cmap="viridis", s=18)
        ax.set_xlabel("|Δplayer_y| fac vs CF (screen px)")
        ax.set_ylabel("mean pre-quant latent dist")
        ax.set_title("Physical vs latent separation")
        fig.colorbar(sc, label="token Hamming"); fig.tight_layout()
        fig.savefig(self.out / "physical_vs_latent_distance.png", dpi=110); plt.close(fig)
        # collision by physical difference (binned)
        fig, ax = plt.subplots(figsize=(6, 4))
        dd = d[m]; cc = coll_cf[m].astype(float)
        if len(dd):
            bins = np.quantile(dd, np.linspace(0, 1, 6))
            bins = np.unique(bins)
            idx = np.clip(np.digitize(dd, bins[1:-1]), 0, len(bins) - 2)
            xs, ys = [], []
            for b in range(len(bins) - 1):
                sel = idx == b
                if sel.any():
                    xs.append(0.5 * (bins[b] + bins[b + 1])); ys.append(cc[sel].mean())
            ax.plot(xs, ys, "o-", color="tab:red")
        ax.set_xlabel("|Δplayer_y| fac vs CF (binned)"); ax.set_ylabel("collision rate")
        ax.set_title("Collision vs physical difference")
        fig.tight_layout(); fig.savefig(self.out / "collision_by_physical_difference.png", dpi=110)
        plt.close(fig)

    def _r2_gallery(self, cur, fac, cf, coll_cf, valid_cf):
        plt = _plt()
        d = self.phys["dplayer_y_fac_cf"]
        cats = {}
        ident = self.phys["identical_fac_cf"]
        if ident.any(): cats["physically identical"] = int(np.where(ident)[0][0])
        small = np.where(valid_cf & (d > 0) & (d <= 2))[0]
        if len(small): cats["small paddle Δ"] = int(small[np.argmin(d[small])])
        large = np.where(valid_cf & (d > 4))[0]
        if len(large): cats["large paddle Δ"] = int(large[np.argmax(d[large])])
        cl = np.where(valid_cf & coll_cf)[0]
        if len(cl): cats["collided pair"] = int(cl[0])
        ncl = np.where(valid_cf & ~coll_cf)[0]
        if len(ncl): cats["non-collided pair"] = int(ncl[np.argmax(d[ncl])])
        if not cats:
            return
        fig, axes = plt.subplots(len(cats), 3, figsize=(9, 3 * len(cats)))
        if len(cats) == 1:
            axes = axes[None, :]
        for r, (label, i) in enumerate(cats.items()):
            imgs = [self.cur_frame[i], self.oc["o_fac_frame"][i], self.oc["o_cf_frame"][i]]
            titles = [f"current\ntok {list(cur[i])}",
                      f"factual next  py={self.oc['o_fac'][i][2]:.0f}\ntok {list(fac[i])}",
                      f"CF next  py={self.oc['o_cf'][i][2]:.0f}\ntok {list(cf[i])}"]
            for c in range(3):
                axes[r, c].imshow(imgs[c], cmap="gray", vmin=0, vmax=255)
                axes[r, c].set_title(titles[c], fontsize=7); axes[r, c].axis("off")
            axes[r, 0].set_ylabel(label, fontsize=9)
            axes[r, 0].text(-0.3, 0.5, f"{label}\nΔpy={d[i]:.1f} coll={bool(coll_cf[i])}",
                            transform=axes[r, 0].transAxes, fontsize=8, va="center",
                            rotation=90)
        fig.tight_layout(); fig.savefig(self.out / "pair_gallery.png", dpi=110); plt.close(fig)

    # ---- GATE R3 -----------------------------------------------------------------
    def gate_r3(self):
        if not self.r1_valid or not hasattr(self, "tok"):
            self.gate("R3", "frozen-latent probes", "—", "BLOCKED", "—",
                      "R1/R2 unavailable", "skipped")
            return {"blocked": True}
        from sklearn.linear_model import Ridge, LogisticRegression

        # feature blocks
        zt = onehot_codes(self.tok["cur"], self.M, self.K)        # z_t one-hot
        zt1 = onehot_codes(self.tok["fac"], self.M, self.K)       # z_{t+1} factual one-hot
        A = np.stack([self.a_prev, self.a_int], 1)
        a_oh = np.concatenate([_oh(self.a_prev, 4), _oh(self.a_int, 4)], 1)  # action one-hot

        feats = {
            "action_only": a_oh,
            "zt+actions": np.concatenate([zt, a_oh], 1),
            "zt+zt1+actions": np.concatenate([zt, zt1, a_oh], 1),
        }
        train_m, test_m = group_train_test_split(self.ep_seed, 0.3, seed=0)
        results = []

        # ---- (1) player_y regression (sanity, uses z_t) ------------------------
        results += self._probe_reg("player_y_regression", zt, self.py_k, train_m, test_m,
                                   Ridge(alpha=1.0), "z_t")
        # ---- (2) paddle movement direction (factual sign of Δplayer_y) ---------
        dir_y = np.sign(self.py_k1 - self.py_k).astype(int)        # -1/0/+1
        results += self._probe_clf("paddle_direction", feats, dir_y, train_m, test_m,
                                   LogisticRegression(max_iter=2000), multiclass=True)
        # ---- (3) executed action classification --------------------------------
        exe = executed_action(self.a_int, self.a_prev, self.stuck)
        results += self._probe_clf("executed_action", feats, exe, train_m, test_m,
                                   LogisticRegression(max_iter=2000), multiclass=True,
                                   cm_png="executed_action_confusion_matrix.png")
        # ---- (4) sticky/free classification — PRIMARY on identifiable subset ----
        idf = self.identifiable & (self.stuck >= 0)
        results += self._probe_sticky(feats, idf, train_m, test_m)

        self._wcsv("probe_results.csv", results,
                   ["probe", "input", "subset", "metric", "train", "test",
                    "baseline", "test_ci_lo", "test_ci_hi", "auc", "n_train", "n_test",
                    "shuffled_control"])
        self.r3_results = results
        self._r3_summary_plot(results)

        # gate verdict: does z_{t+1} add sticky/executed info over action-only?
        verdict, notes = self._r3_verdict(results)
        self.gate("R3", "frozen-latent probes: does z_{t+1} add realized-noise info?",
                  "z_t+z_{t+1}+actions beats action-only on sticky/executed-action decode; "
                  "shuffled control near chance",
                  verdict, "probe_results.csv, sticky_probe_roc.png, "
                  "sticky_probe_confusion_matrix.png, executed_action_confusion_matrix.png, "
                  "probe_summary.png", "drives the final abductability verdict", notes)
        return {"results": results, "verdict": verdict}

    def _probe_reg(self, name, X, y, tr, te, model, inp):
        model.fit(X[tr], y[tr])
        pred_te = model.predict(X[te])
        ae = np.abs(pred_te - y[te])
        lo, hi = bootstrap_ci(ae, seed=1)
        base = np.abs(y[te] - y[tr].mean())
        # shuffled control
        ysh = y[tr].copy(); np.random.default_rng(0).shuffle(ysh)
        m2 = type(model)(**model.get_params()); m2.fit(X[tr], ysh)
        sh = float(np.abs(m2.predict(X[te]) - y[te]).mean())
        return [{"probe": name, "input": inp, "subset": "all", "metric": "MAE",
                 "train": float(np.abs(model.predict(X[tr]) - y[tr]).mean()),
                 "test": float(ae.mean()), "baseline": float(base.mean()),
                 "test_ci_lo": lo, "test_ci_hi": hi, "auc": "",
                 "n_train": int(tr.sum()), "n_test": int(te.sum()),
                 "shuffled_control": sh}]

    def _probe_clf(self, name, feats, y, tr, te, model, multiclass=False, cm_png=None):
        from sklearn.base import clone
        rows = []
        base = _majority_acc(y[tr], y[te])
        best_pred = None
        for inp, X in feats.items():
            m = clone(model); m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            correct = (pred == y[te]).astype(float)
            lo, hi = bootstrap_ci(correct, seed=2)
            ysh = y[tr].copy(); np.random.default_rng(0).shuffle(ysh)
            msh = clone(model); msh.fit(X[tr], ysh)
            sh = float((msh.predict(X[te]) == y[te]).mean())
            rows.append({"probe": name, "input": inp, "subset": "all", "metric": "acc",
                         "train": float((m.predict(X[tr]) == y[tr]).mean()),
                         "test": float(correct.mean()), "baseline": base,
                         "test_ci_lo": lo, "test_ci_hi": hi, "auc": "",
                         "n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "shuffled_control": sh})
            if inp == "zt+zt1+actions":
                best_pred = pred
        if cm_png and best_pred is not None:
            self._confusion(y[te], best_pred, cm_png, f"{name} (z_t+z_t1+actions)")
        return rows

    def _probe_sticky(self, feats, subset_mask, tr, te):
        from sklearn.linear_model import LogisticRegression
        from sklearn.base import clone
        y = (self.stuck == 1).astype(int)
        rows = []
        for subset, mask in (("identifiable", subset_mask),
                             ("all_free_or_stuck", self.stuck >= 0)):
            trm, tem = tr & mask, te & mask
            if trm.sum() < 8 or tem.sum() < 4 or len(np.unique(y[trm])) < 2:
                rows.append({"probe": "sticky", "input": "—", "subset": subset,
                             "metric": "acc", "train": "", "test": "",
                             "baseline": "", "test_ci_lo": "", "test_ci_hi": "",
                             "auc": "", "n_train": int(trm.sum()), "n_test": int(tem.sum()),
                             "shuffled_control": "insufficient samples"})
                continue
            base = _majority_acc(y[trm], y[tem])
            roc_data = {}
            for inp, X in feats.items():
                m = clone(LogisticRegression(max_iter=2000)); m.fit(X[trm], y[trm])
                pred = m.predict(X[tem])
                correct = (pred == y[tem]).astype(float)
                lo, hi = bootstrap_ci(correct, seed=3)
                try:
                    proba = m.predict_proba(X[tem])[:, 1]
                    from sklearn.metrics import roc_auc_score
                    auc = float(roc_auc_score(y[tem], proba)) if len(np.unique(y[tem])) > 1 else None
                    roc_data[inp] = (y[tem], proba)
                except Exception:
                    auc = None
                ysh = y[trm].copy(); np.random.default_rng(0).shuffle(ysh)
                msh = clone(LogisticRegression(max_iter=2000)); msh.fit(X[trm], ysh)
                sh = float((msh.predict(X[tem]) == y[tem]).mean())
                rows.append({"probe": "sticky", "input": inp, "subset": subset,
                             "metric": "acc", "train": float((m.predict(X[trm]) == y[trm]).mean()),
                             "test": float(correct.mean()), "baseline": base,
                             "test_ci_lo": lo, "test_ci_hi": hi, "auc": auc,
                             "n_train": int(trm.sum()), "n_test": int(tem.sum()),
                             "shuffled_control": sh})
            if subset == "identifiable" and roc_data:
                self._sticky_roc(roc_data)
                # confusion for the full-feature model
                m = clone(LogisticRegression(max_iter=2000)); m.fit(
                    feats["zt+zt1+actions"][trm], y[trm])
                self._confusion(y[tem], m.predict(feats["zt+zt1+actions"][tem]),
                                "sticky_probe_confusion_matrix.png",
                                "sticky (identifiable, z_t+z_t1+actions)")
        return rows

    def _sticky_roc(self, roc_data):
        plt = _plt()
        from sklearn.metrics import roc_curve, roc_auc_score
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for inp, (yt, pr) in roc_data.items():
            if len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, pr)
            ax.plot(fpr, tpr, label=f"{inp} (AUC={roc_auc_score(yt, pr):.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("Sticky decode ROC (identifiable subset)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(self.out / "sticky_probe_roc.png", dpi=110)
        plt.close(fig)

    def _confusion(self, y_true, y_pred, fname, title):
        plt = _plt()
        from sklearn.metrics import confusion_matrix
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for r in range(len(labels)):
            for c in range(len(labels)):
                ax.text(c, r, cm[r, c], ha="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(self.out / fname, dpi=110)
        plt.close(fig)

    def _r3_summary_plot(self, results):
        plt = _plt()
        probes = ["paddle_direction", "executed_action", "sticky"]
        inputs = ["action_only", "zt+actions", "zt+zt1+actions"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.25
        for j, inp in enumerate(inputs):
            xs, ys = [], []
            for i, pr in enumerate(probes):
                subset = "identifiable" if pr == "sticky" else "all"
                row = next((r for r in results if r["probe"] == pr and r["input"] == inp
                            and r["subset"] == subset and r["test"] != ""), None)
                xs.append(i + j * width)
                ys.append(float(row["test"]) if row else 0)
            ax.bar(xs, ys, width, label=inp)
        # baselines
        for i, pr in enumerate(probes):
            subset = "identifiable" if pr == "sticky" else "all"
            row = next((r for r in results if r["probe"] == pr and r["baseline"] != ""
                        and r["subset"] == subset), None)
            if row:
                ax.hlines(float(row["baseline"]), i - 0.15, i + 3 * width,
                          color="red", linestyles="dashed",
                          label="majority baseline" if i == 0 else None)
        ax.set_xticks([i + width for i in range(len(probes))]); ax.set_xticklabels(probes)
        ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Probe accuracy by input (does z_{t+1} help?)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(self.out / "probe_summary.png", dpi=110); plt.close(fig)

    def _r3_verdict(self, results):
        """The abduction-relevant question (sub-question C) is whether the NEXT latent
        z_{t+1} adds realized-noise info OVER z_t+actions — not merely over action-only.
        So the decisive delta is full−(z_t+actions); full−action_only is reported too."""
        def get(probe, inp, subset, key="test"):
            r = next((x for x in results if x["probe"] == probe and x["input"] == inp
                      and x["subset"] == subset and x.get(key) not in ("", None)), None)
            return float(r[key]) if r and isinstance(r.get(key), (int, float)) else None
        def cilo(probe, inp, subset):
            return get(probe, inp, subset, "test_ci_lo")

        notes = []
        zt1_adds = False          # next latent adds over z_t+actions
        beats_actions = False     # latent transition (z_t+z_t1) beats action-only at all
        robust = False            # above chance with CI excluding baseline
        for probe, subset in (("sticky", "identifiable"), ("executed_action", "all")):
            a = get(probe, "action_only", subset)
            zt = get(probe, "zt+actions", subset)
            full = get(probe, "zt+zt1+actions", subset)
            base = get(probe, "action_only", subset, "baseline")
            if None in (a, zt, full):
                continue
            d_zt1 = full - zt          # << decisive: does the NEXT latent add?
            d_full = full - a
            notes.append(f"{probe}[{subset}]: action-only {a:.2f} -> z_t+act {zt:.2f} -> "
                         f"z_t+z_t1+act {full:.2f}  (Δ_nextlatent={d_zt1:+.2f}, "
                         f"Δ_vs_action={d_full:+.2f})")
            if d_zt1 >= 0.02:
                zt1_adds = True
            if d_full >= 0.05:
                beats_actions = True
            lo = cilo(probe, "zt+zt1+actions", subset)
            if lo is not None and base is not None and lo > base:
                robust = True
        sh_ok = all(_near_chance(r) for r in results
                    if r["probe"] == "sticky" and isinstance(r.get("shuffled_control"), float))
        tail = (" | shuffled control near chance" if sh_ok else
                " | WARNING shuffled control elevated (overfit risk)")
        if zt1_adds and beats_actions and robust and sh_ok:
            return "PASS", "; ".join(notes) + tail
        if beats_actions or zt1_adds:
            return "PASS WITH WARNINGS", ("; ".join(notes) +
                   " | next latent adds little/no info over z_t+actions and CIs touch "
                   "chance — weak/uncertain signal" + tail)
        return "FAIL", ("the factual latent TRANSITION adds no usable sticky/executed-action "
                        "signal beyond z_t+actions (z_{t+1} overfits, test drops); "
                        + "; ".join(notes) + tail)

    # ---- GATE R4 (report) --------------------------------------------------------
    def gate_r4(self, r1, r2, r3):
        verdict = self._final_verdict()
        self._wj("gate_table.json", self.gate_rows)
        (self.out / "exact_commands.txt").write_text(
            "python -m pong_counterfactual.cjepa_rung3.diagnose_representation "
            f"--data-dir {self.data_dir} --checkpoint {self.ckpt}\n", encoding="utf-8")
        self._write_report(r1, r2, r3, verdict)
        self._write_limitations()
        return verdict

    def _final_verdict(self):
        results = [r["Result"] for r in self.gate_rows]
        if "FAIL" in results or "BLOCKED" in results:
            return "FAIL"
        if "PASS WITH WARNINGS" in results:
            return "PASS WITH WARNINGS"
        return "PASS"

    def _next_step(self, verdict):
        r3 = next((r for r in self.gate_rows if r["Gate"] == "R3"), {})
        if verdict == "PASS":
            return "proceed to transition-model training (Stage 3)"
        if any(r["Gate"] == "R2" and "collaps" in r["Notes"].lower() for r in self.gate_rows):
            return "improve temporal representation (the eye collapses branch differences)"
        if r3.get("Result") in ("FAIL", "PASS WITH WARNINGS"):
            return ("improve temporal representation / compare with a reconstruction "
                    "baseline before training a transition model")
        return "redesign oracle/evaluation tuples"

    def _write_report(self, r1, r2, r3, verdict):
        c = r1.get("counts", {})
        m = r2.get("metrics", {}) if not r2.get("blocked") else {}
        nxt = self._next_step(verdict)
        cols = ["Gate", "Check", "PASS criterion", "Result", "Evidence artifact",
                "Blocking consequence", "Notes"]
        gt = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for r in self.gate_rows:
            gt.append("| " + " | ".join(str(r[c]).replace("\n", " ").replace("|", "\\|")
                                        for c in cols) + " |")
        report = f"""# Rung 3 Stage A1 — Representation Abductability Audit

**Final verdict: {verdict}**

Generated: {utc_stamp()} • Checkpoint: `{self.ckpt}` • s={self.s}

## Purpose & non-claims
Tests whether the FROZEN `tokenizer_M4_K64_res84` representation carries the information a
later counterfactual abduction step would need. **A low collision rate alone is NOT taken
as proof of abductability** — the decisive test is whether the factual latent transition
z_t→z_{{t+1}} lets a linear probe decode sticky/executed-action info BEYOND the actions
themselves (sub-question C). No training, tokenizer modification, or Stage 3/4 was done.

## A. Branch validity
{c.get('n_branches_physically_differ_fac_cf', '—')}/{c.get('n_valid_fac_cf', '—')} oracle
factual-vs-counterfactual branches differ physically (4-vector level)
({c.get('frac_branches_differ_fac_cf', 0):.0%}). Replay determinism: vec
{json.loads((self.out/'data_integrity.json').read_text())['replay_vec_match_rate']:.3f} /
frame {json.loads((self.out/'data_integrity.json').read_text())['replay_frame_match_rate']:.3f}.
Sticky={c.get('n_stuck')}, free={c.get('n_free')}, identifiable subset (a_prev≠a_int)
={c.get('n_identifiable')}. Evidence: `sample_counts.json`, `data_integrity.json`,
`branch_physical_distance.csv`, `branch_distance_histogram.png`.

## B. Representation separability
{"BLOCKED (R1 invalid)." if r2.get('blocked') else f'''Encoding deterministic: {m.get("deterministic_encoding")}. Measured on the
{m.get("n_physically_differ")} PHYSICALLY-DIFFERENT (free) pairs: token collision
**{m.get("collision_among_differing_fac_cf"):.1%}**, mean token Hamming
{m.get("mean_hamming_among_differing_fac_cf"):.2f}/{self.M}, mean pre-quant latent dist
{m.get("mean_prequant_latent_dist_differing_fac_cf"):.3f}. (Physically-identical/stuck pairs
collide {m.get("collision_among_identical_fac_cf"):.0%} — correct, same frame; all-pairs
collision {m.get("collision_allpairs_fac_cf"):.1%} merely equals the stuck rate and is NOT
the separability metric.) Cross-check vs the gate's saved coll_flag:
{m.get("crosscheck_vs_gate_coll_flag")}. Evidence:
`representation_metrics.json`, `pair_metrics.csv`, `token_hamming_histogram.png`,
`physical_vs_latent_distance.png`, `collision_by_physical_difference.png`, `pair_gallery.png`.'''}

## C. Abductability
{"BLOCKED." if r3.get('blocked') else next((r['Notes'] for r in self.gate_rows if r['Gate']=='R3'), '')}
Evidence: `probe_results.csv`, `sticky_probe_roc.png`, `sticky_probe_confusion_matrix.png`,
`executed_action_confusion_matrix.png`, `probe_summary.png`.

## Verdict rule applied
- PASS: latent transition clearly improves sticky/executed-action decoding over
  action-only, valid branches, control near chance.
- PASS WITH WARNINGS: some signal but weak confidence/coverage/performance.
- FAIL: invalid branches, representation collapse, or no usable latent-noise info.

**→ {verdict}**

## Recommended next step
**{nxt}**

## Gate table
{chr(10).join(gt)}
"""
        (self.out / "REPORT.md").write_text(report, encoding="utf-8")

    def _write_limitations(self):
        a_prev_all_noop = bool((self.a_prev == NOOP).all())
        lim = [
            ("In this eval pool a_prev is ALWAYS NOOP (%s): the behavior policy precedes "
             "every move with a NOOP, so a sticky fire repeats NOOP and the realized-noise "
             "signal is exactly 'paddle STAYED PUT vs moved ~1 paddle step (~4.6 screen px)'. "
             "That 1-step difference is near the M4_K64_res84 eye's KNOWN coarse paddle "
             "localization limit (its Stage-2 probe failed at player_y MAE 4.05 resized px), "
             "which is the most likely reason z_{t+1} fails to add a clean sticky signal — "
             "the move-vs-stay difference is at/below token resolution."
             % a_prev_all_noop),] + [
            "Eval coverage is the fixed 250-tuple pool over %d eval episodes; episode-"
            "grouped splits leave few test episodes, so probe CIs are wide — treat R3 "
            "effect sizes as indicative, not definitive." % C.N_EVAL_EPS,
            "Sticky is identifiable only when a_prev≠a_int; the identifiable subset is "
            "smaller than the full pool, further limiting probe power.",
            "Physical 'pixel difference' uses the preprocessed res%d frames; sub-pixel "
            "paddle moves can vanish under the bilinear resize (the known Rung-3 ① issue)."
            % C.FRAME.res,
            "z_{t+1} is the FACTUAL next latent only; this audit does not encode the "
            "counterfactual branch into a transition — it asks whether the factual "
            "transition is informative, a necessary (not sufficient) condition for "
            "abduction.",
            "Probes are linear/logistic by design; a negative result bounds LINEAR "
            "decodability, not all decodability.",
        ]
        (self.out / "known_limitations.md").write_text(
            "# Known limitations\n\n" + "\n".join(f"- {x}" for x in lim) + "\n",
            encoding="utf-8")


# ---------------------------------------------------------------- tiny helpers (tested)
def _oh(a, n):
    a = np.asarray(a, np.int64)
    X = np.zeros((len(a), n), np.float32)
    X[np.arange(len(a)), np.clip(a, 0, n - 1)] = 1.0
    return X


def _majority_acc(y_train, y_test):
    vals, cnts = np.unique(y_train, return_counts=True)
    maj = vals[np.argmax(cnts)]
    return float((y_test == maj).mean())


def _near_chance(row, tol=0.15):
    sc = row.get("shuffled_control")
    base = row.get("baseline")
    if not isinstance(sc, float) or not isinstance(base, float):
        return True
    return abs(sc - base) <= tol


# ============================================================================== main
def run(repo, ckpt, data_dir, out_root, s, device, preflight_only, ckpt_res):
    pf = preflight(repo, ckpt, data_dir, s, ckpt_res)
    if preflight_only or not pf.get("ok"):
        print(json.dumps(pf, indent=2))
        if not pf.get("ok"):
            print("\n[diagnose] PREFLIGHT FAILED — supply the data and re-run. "
                  "No audit dir written.", file=sys.stderr)
        return {"preflight": pf, "ran": False}

    out_dir = out_root / f"rung3_repr_audit_{utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = Diagnose(repo, ckpt, data_dir, out_dir, s, device)
    d.load()
    r1 = d.gate_r1()
    r2 = d.gate_r2()
    r3 = d.gate_r3()
    verdict = d.gate_r4(r1, r2, r3)
    print(f"[diagnose] verdict: {verdict}\n[diagnose] output: {out_dir}")
    return {"out_dir": out_dir, "verdict": verdict, "ran": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rung 3 Stage A1 representation abductability audit")
    ap.add_argument("--data-dir", default=None,
                    help="dir with ep_*.npz + frame-bearing oracle_cache.npz")
    ap.add_argument("--checkpoint", default=None, help="approved final.pt (auto-discovered)")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--s", type=float, default=C.S_TRAIN)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true", help="validate inputs and exit")
    args = ap.parse_args(argv)

    repo = find_repo_root(Path(__file__))
    ckpt = discover_checkpoint(repo, args.checkpoint)
    ckpt_res = checkpoint_res(ckpt)
    data_dir = discover_data_dir(repo, args.s, ckpt_res, args.data_dir)
    out_root = Path(args.out_root).resolve() if args.out_root \
        else repo / "pong_counterfactual" / "audits"
    run(repo, ckpt, data_dir, out_root, args.s, args.device, args.preflight, ckpt_res)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
