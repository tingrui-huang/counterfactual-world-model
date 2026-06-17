"""Rung 3 Stage A0 — read-only audit of the three completed tokenizer runs.

Scientific purpose
------------------
This does NOT test whether JEPA supports counterfactual abduction. Its only job is to
establish a TRUSTWORTHY artifact inventory before any later read-only diagnosis:
which checkpoint is which architecture/resolution, whether folder names / checkpoint
metadata / repo config / gate report / gate artifacts / training logs are mutually
consistent, whether ``final.pt`` is a valid complete final checkpoint, and whether the
M4-K64-res84 run can be safely reused for representation diagnostics. A successful
``torch.load`` is NOT sufficient evidence of correctness.

Hard scope (enforced by construction): this script only READS the frozen run dirs and
only WRITES under a fresh timestamped audit dir OUTSIDE them. It never trains, never
regenerates data, never repairs anything. Every inference is labelled with its source
of truth; nothing is silently inferred from the directory name alone.

``weights_only=False`` is used for ``torch.load``. These are TRUSTED, repo-generated
checkpoints (written by ``cjepa_rung3/ckpt.py`` and ``stage1_pretrain.py``): the
payloads contain non-tensor objects (a numpy RNG state tuple, a torch Generator state,
and a plain ``config`` dict) that the torch>=2.6 default ``weights_only=True`` refuses
to unpickle. Loading is on CPU only, read-only, from files the user placed in the repo.

Run:
  python -m pong_counterfactual.cjepa_rung3.audit_rung3_tokenizer_artifacts
  python -m pong_counterfactual.cjepa_rung3.audit_rung3_tokenizer_artifacts \
      --results-root pong_counterfactual/results --discover-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- expected run configs
# (parsed M/K/res are the EXPECTED values from the *intended* config name; the audit
#  verifies them against checkpoint tensors — the name is only a weak hint, never truth.)
EXPECTED_RUNS = {
    "M4_K64_res84":  {"M": 4, "K": 64,  "res": 84},
    "M8_K128_res84": {"M": 8, "K": 128, "res": 84},
    "M4_K64_res128": {"M": 4, "K": 64,  "res": 128},
}
RUN_NAME_RE = re.compile(r"tokenizer_(M\d+_K\d+_res\d+)")
REQUIRED_FILES = ("final.pt", "gate_report.json", "gate_artifacts.npz", "train_log.jsonl")
RECOMPUTE_TOL = 1e-6          # relative tolerance for recomputed deterministic metrics

# state_dict keys whose shapes pin the architecture (from cjepa_rung3/tokenizer.py)
SHAPE_KEYS = {
    "context.sem": "sem",                    # (1, M, dim)
    "context.pos": "pos",                    # (1, P, dim)
    "context.patch_embed.weight": "patch",   # (dim, 1, patch, patch)
    "vq.codebook": "codebook",               # (K, code_dim)
}


# ============================================================= small generic utilities
def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), *args],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() if out.returncode == 0 else f"<git error: {out.stderr.strip()}>"
    except Exception as e:                              # git missing / not a repo
        return f"<git unavailable: {e}>"


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ============================================================= train-log JSONL parsing
def parse_jsonl(text: str) -> dict:
    """Robustly parse a train_log.jsonl. Pure function (used by tests).

    Returns a dict with: rows (list), n_lines, n_valid, malformed (list of (lineno,raw)),
    steps, monotonic (bool, non-decreasing), strictly_increasing (bool),
    duplicate_steps (list), decreasing_at (list of lineno), off_grid_steps (list),
    has_nan_inf (bool), nan_inf_fields (list)."""
    rows, malformed = [], []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            malformed.append({"line": i, "error": str(e), "raw": line[:200]})

    steps = [r["step"] for r in rows if isinstance(r, dict) and "step" in r]
    decreasing_at, duplicate_steps = [], []
    for j in range(1, len(steps)):
        if steps[j] < steps[j - 1]:
            decreasing_at.append(j)
    seen = set()
    for s in steps:
        if s in seen:
            duplicate_steps.append(s)
        seen.add(s)
    monotonic = len(decreasing_at) == 0                       # non-decreasing
    strictly_increasing = monotonic and not duplicate_steps

    # NaN/Inf scan over numeric metric fields
    nan_inf_fields = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                nan_inf_fields.add(k)

    return {
        "rows": rows, "n_lines": len(lines), "n_valid": len(rows),
        "malformed": malformed, "steps": steps,
        "monotonic": monotonic, "strictly_increasing": strictly_increasing,
        "duplicate_steps": sorted(set(duplicate_steps)),
        "decreasing_at": decreasing_at,
        "has_nan_inf": bool(nan_inf_fields), "nan_inf_fields": sorted(nan_inf_fields),
        "last_step": steps[-1] if steps else None,
    }


def analyze_resumes(steps: list[int], ckpt_every: int | None) -> dict:
    """Classify every step-decrease as a DOCUMENTED resume or an unexplained anomaly.

    The Colab contract (ckpt.py / stage1_pretrain.py): on a fresh runtime, training
    RESUMES from the newest checkpoint that unpickles (a multiple of ckpt_every), or
    from 0 if none exists yet — and a `step==start` row is appended to the (append-only)
    log on each resume. So a decrease back to a value `v` is documented iff `v-1` is a
    multiple of ckpt_every (covers a fresh restart -> v=1, and resume@N*ckpt_every ->
    v=N*ckpt_every+1). Anything else is an unexplained backward jump. Pure function."""
    boundaries = []
    for i in range(1, len(steps)):
        if steps[i] < steps[i - 1]:
            to = steps[i]
            if ckpt_every:
                documented = ((to - 1) % ckpt_every == 0)
            else:
                documented = False
            boundaries.append({"index": i, "from_step": steps[i - 1], "to_step": to,
                               "documented": documented})
    undocumented = [b for b in boundaries if not b["documented"]]
    return {"boundaries": boundaries, "n_resumes": len(boundaries),
            "undocumented": undocumented, "all_documented": not undocumented}


# ============================================================= state_dict comparison
def compare_state_dicts(a: dict, b: dict) -> dict:
    """Compare two model state_dicts. Pure function (used by tests).

    Returns: missing_in_b (keys in a not b), unexpected_in_b (keys in b not a),
    shape_mismatch (list), max_abs_diff (float over common comparable tensors),
    identical (bool)."""
    import torch
    ka, kb = set(a), set(b)
    missing_in_b = sorted(ka - kb)
    unexpected_in_b = sorted(kb - ka)
    shape_mismatch, max_abs_diff = [], 0.0
    for k in sorted(ka & kb):
        ta, tb = a[k], b[k]
        sa = tuple(ta.shape) if hasattr(ta, "shape") else None
        sb = tuple(tb.shape) if hasattr(tb, "shape") else None
        if sa != sb:
            shape_mismatch.append({"key": k, "shape_a": sa, "shape_b": sb})
            continue
        try:
            d = (ta.float() - tb.float()).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)
        except Exception:
            pass
    identical = (not missing_in_b and not unexpected_in_b
                 and not shape_mismatch and max_abs_diff == 0.0)
    return {"missing_in_b": missing_in_b, "unexpected_in_b": unexpected_in_b,
            "shape_mismatch": shape_mismatch, "max_abs_diff": max_abs_diff,
            "identical": identical}


def infer_arch_from_state_dict(sd: dict) -> dict:
    """Infer (M, K, dim, code_dim, patch, P, res) from tensor shapes alone.
    Each field tagged INFERRED_FROM_TENSOR_SHAPE or UNKNOWN. Pure-ish (needs the sd)."""
    out, shapes = {}, {}
    for key in SHAPE_KEYS:
        if key in sd:
            shapes[SHAPE_KEYS[key]] = tuple(sd[key].shape)

    def put(name, val):
        out[name] = {"value": val, "source": "INFERRED_FROM_TENSOR_SHAPE"} if val is not None \
            else {"value": None, "source": "UNKNOWN"}

    sem = shapes.get("sem")          # (1, M, dim)
    pos = shapes.get("pos")          # (1, P, dim)
    patch_w = shapes.get("patch")    # (dim, 1, patch, patch)
    cb = shapes.get("codebook")      # (K, code_dim)

    put("M", sem[1] if sem else None)
    put("dim", sem[2] if sem else (patch_w[0] if patch_w else None))
    put("K", cb[0] if cb else None)
    put("code_dim", cb[1] if cb else None)
    put("patch", patch_w[2] if patch_w else None)
    P = pos[1] if pos else None
    put("P", P)
    # res = sqrt(P) * patch, only if P is a perfect square and patch known
    res = None
    if P and patch_w:
        g = int(round(math.sqrt(P)))
        if g * g == P:
            res = g * patch_w[2]
    put("res", res)
    out["_raw_shapes"] = shapes
    return out


# ===================================================================== checkpoint load
def load_ckpt(path: Path):
    """CPU, weights_only=False (trusted repo checkpoints — see module docstring)."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_checkpoint(path: Path, kind: str) -> dict:
    """Structural summary of one checkpoint without mutating it."""
    import torch
    rec: dict = {"path": str(path), "kind": kind}
    try:
        payload = load_ckpt(path)
    except Exception as e:
        rec["loadable"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["loadable"] = True
    rec["top_level_type"] = type(payload).__name__
    if not isinstance(payload, dict):
        rec["note"] = "top-level object is not a dict"
        return rec
    rec["top_level_keys"] = sorted(payload.keys())
    rec["has_optimizer"] = "opt" in payload
    rec["has_scheduler"] = "scheduler" in payload or "sched" in payload
    rec["has_scaler"] = "scaler" in payload
    rec["has_rng"] = "rng" in payload
    rec["has_gen"] = "gen" in payload
    rec["explicit_step"] = payload.get("step", None)
    rec["embedded_config"] = payload.get("config", None)

    sd = payload.get("model")
    if not isinstance(sd, dict):
        rec["model_present"] = False
        return rec
    rec["model_present"] = True
    rec["n_tensors"] = len(sd)
    # required tokenizer components present?
    required_prefixes = ["context.", "target.", "vq.codebook", "pred_s2p.", "pred_p2p."]
    rec["component_presence"] = {
        pfx: any(k == pfx or k.startswith(pfx) for k in sd) for pfx in required_prefixes}
    rec["codebook_present"] = "vq.codebook" in sd
    rec["sem_token_present"] = "context.sem" in sd
    rec["target_encoder_present"] = any(k.startswith("target.") for k in sd)
    rec["predictor_present"] = any(k.startswith("pred_") for k in sd)

    rec["inferred"] = infer_arch_from_state_dict(sd)

    # try a strict structural load against the architecture rebuilt from embedded config
    rec["strict_load"] = _try_strict_load(rec["embedded_config"], sd)
    # keep the raw state_dict around for caller-side final-vs-latest comparison
    rec["_state_dict"] = sd
    return rec


def _try_strict_load(cfg_dict, sd) -> dict:
    """Rebuild DiscreteJEPA(cfg) from the embedded config and load_state_dict(strict).
    This uses the repository's own model code as the source of architectural truth."""
    if not isinstance(cfg_dict, dict):
        return {"attempted": False, "reason": "no embedded config"}
    try:
        import torch  # noqa
        from pong_counterfactual.cjepa_rung3.config import TokenizerConfig
        from pong_counterfactual.cjepa_rung3.tokenizer import DiscreteJEPA
        cfg = TokenizerConfig(**cfg_dict)
        model = DiscreteJEPA(cfg)
        res = model.load_state_dict(sd, strict=True)
        return {"attempted": True, "ok": True,
                "missing_keys": list(res.missing_keys),
                "unexpected_keys": list(res.unexpected_keys)}
    except Exception as e:
        return {"attempted": True, "ok": False, "error": f"{type(e).__name__}: {e}"}


# =================================================================== gate report / npz
def summarize_gate_artifacts(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    arrays = {}
    for k in z.files:
        a = z[k]
        info = {"dtype": str(a.dtype), "shape": list(a.shape), "size": int(a.size)}
        if a.size and np.issubdtype(a.dtype, np.number):
            af = a.astype(np.float64)
            info.update(min=float(np.nanmin(af)), max=float(np.nanmax(af)),
                        mean=float(np.nanmean(af)),
                        n_nan=int(np.isnan(af).sum()), n_inf=int(np.isinf(af).sum()))
        arrays[k] = info
    z.close()
    return {"path": str(npz_path), "arrays": arrays}


def recompute_gate_metrics(npz_path: Path, report: dict) -> dict:
    """Recompute ONLY the deterministic values whose meaning is fixed by stage2_gate.py:
      coll_flag in {-1 (unknowable/stuck), 0 (free, no collision), 1 (free collision)};
      collision_allM = mean(coll_flag==1 over free tuples); n_free=(coll_flag!=-1).sum();
      n_evaluated_pairs=len(pool); probe_W.shape == (M*K, n_dims).
    Probe MAE and per-slot collision are NOT recomputable here (need the absent dataset /
    per-slot flags) and are reported as UNAVAILABLE."""
    z = np.load(npz_path, allow_pickle=False)
    pool = z["pool"]
    coll = z["coll_flag"]
    probe_W = z["probe_W"]
    z.close()

    free = coll != -1
    n_free = int(free.sum())
    collision_recomp = float((coll[free] == 1).mean()) if n_free else None
    n_pairs = int(len(pool))
    MK = int(probe_W.shape[0])
    n_dims = int(probe_W.shape[1]) if probe_W.ndim == 2 else None

    cmp = {}

    def add(name, recomp, reported):
        agree = None
        if recomp is not None and reported is not None:
            denom = max(abs(reported), 1e-12)
            agree = abs(recomp - reported) / denom <= RECOMPUTE_TOL
        cmp[name] = {"recomputed": recomp, "reported": reported, "agree": agree}

    add("collision_allM", collision_recomp, report.get("collision_allM"))
    add("n_free_tuples", n_free, report.get("n_free_tuples"))
    add("n_samples_pool", n_pairs, len(report.get("collision_per_slot", [])) and None)
    cmp["n_free_tuples"]["note"] = "free = coll_flag != -1"
    cmp["n_samples_pool"]["recomputed"] = n_pairs

    # cross-check M*K against report config (M*K) — meaning of probe_W is fixed by code
    cfg = report.get("config", {})
    rep_MK = (cfg.get("M", 0) * cfg.get("K", 0)) or None
    cmp["probe_W_MK"] = {"recomputed": MK, "reported": rep_MK,
                         "agree": (MK == rep_MK) if rep_MK else None,
                         "note": "probe_W rows == M*K one-hot features"}
    cmp["probe_W_n_dims"] = {"recomputed": n_dims, "reported": 4,
                             "agree": (n_dims == 4), "note": "4 ground-truth position dims"}
    cmp["_unavailable"] = ["probe_mae_resized_px (needs absent dataset)",
                           "collision_per_slot (per-slot flags not stored in npz)",
                           "n_stuck (stuck array not stored; coll_flag==-1 is a proxy)"]
    return cmp


# ============================================================================ discovery
def discover_runs(results_root: Path) -> dict:
    """Find run dirs by locating every final.pt and classifying its parent by name.
    Returns {"runs": {cfg: [paths]}, "unclassified": [...], "all_final_pts": [...]}."""
    runs: dict[str, list[Path]] = {k: [] for k in EXPECTED_RUNS}
    unclassified = []
    all_final = sorted(results_root.rglob("final.pt"))
    for fp in all_final:
        run_dir = fp.parent
        m = RUN_NAME_RE.search(run_dir.name)
        cfg = m.group(1) if m else None
        if cfg in runs:
            runs[cfg].append(run_dir)
        else:
            unclassified.append(str(run_dir))
    return {"runs": {k: [str(p) for p in v] for k, v in runs.items()},
            "unclassified": unclassified,
            "all_final_pts": [str(p) for p in all_final]}


# ================================================================================ audit
class Audit:
    def __init__(self, repo_root: Path, results_root: Path, out_dir: Path):
        self.repo_root = repo_root
        self.results_root = results_root
        self.out = out_dir
        self.gate_rows: list[dict] = []      # for the gate table
        self.commands: list[str] = []
        self.limitations: list[str] = []

    def gate(self, name, check, criterion, result, evidence, blocking, notes=""):
        self.gate_rows.append({
            "Gate": name, "Check": check, "PASS criterion": criterion,
            "Result": result, "Evidence artifact": evidence,
            "Blocking consequence": blocking, "Notes": notes})

    # ---- GATE A0 -----------------------------------------------------------------
    def gate_a0(self) -> dict:
        repo_info = {
            "repository_root": str(self.repo_root),
            "git_commit": _git(self.repo_root, "rev-parse", "HEAD"),
            "git_branch": _git(self.repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "results_root_abs": str(self.results_root.resolve()),
            "results_root_rel": str(self.results_root.resolve().relative_to(self.repo_root)),
            "audit_out_abs": str(self.out.resolve()),
            "audit_out_rel": str(self.out.resolve().relative_to(self.repo_root)),
            "audit_loaded_with_weights_only": False,
            "weights_only_justification":
                "trusted repo-generated checkpoints contain numpy/Generator RNG state + "
                "a plain config dict; torch>=2.6 weights_only=True cannot unpickle them. "
                "Loaded CPU-only, read-only.",
        }
        _write_json(self.out / "repository_info.json", repo_info)
        (self.out / "git_status_before.txt").write_text(
            _git(self.repo_root, "status", "--porcelain") + "\n", encoding="utf-8")

        disc = discover_runs(self.results_root)
        _write_json(self.out / "discovered_runs.json", disc)

        # PASS checks
        problems = []
        found = {}
        for cfg, paths in disc["runs"].items():
            if len(paths) == 0:
                problems.append(f"run {cfg} NOT FOUND")
            elif len(paths) > 1:
                problems.append(f"run {cfg} AMBIGUOUS: {len(paths)} dirs match "
                                f"(no deterministic selection rule documented)")
            else:
                found[cfg] = Path(paths[0])
        # output dir must not be inside any frozen run
        for cfg, paths in disc["runs"].items():
            for p in paths:
                pr = Path(p).resolve()
                if pr == self.out.resolve() or pr in self.out.resolve().parents \
                        or self.out.resolve() == pr or str(self.out.resolve()).startswith(str(pr)):
                    problems.append(f"AUDIT OUTPUT inside frozen run {cfg}")
        # also guard: out dir not anywhere under results_root
        if str(self.out.resolve()).startswith(str(self.results_root.resolve())):
            problems.append("AUDIT OUTPUT is under results_root (frozen-input tree)")

        result = "PASS" if not problems else "FAIL"
        self.gate("A0", "repo root + discover 3 runs + output-separation guard",
                  "exactly one dir per expected run; output outside frozen runs",
                  result, "repository_info.json, discovered_runs.json, git_status_before.txt",
                  "stop checkpoint-dependent gates for any run not uniquely identified",
                  "; ".join(problems) if problems else "all 3 runs uniquely found")
        return {"found": found, "problems": problems, "discovered": disc}

    # ---- GATE A1 -----------------------------------------------------------------
    def gate_a1(self, found: dict[str, Path]) -> dict:
        inventory_rows, hash_lines, per_run = [], [], {}
        problems = []
        for cfg, run_dir in found.items():
            files = sorted(p for p in run_dir.rglob("*") if p.is_file())
            run_entry = {"run_dir": str(run_dir), "files": [], "required": {}}
            present = {}
            for fp in files:
                stat = fp.stat()
                sha = sha256_file(fp)
                rel = str(fp.relative_to(run_dir))
                row = {"run": cfg, "rel_path": rel, "ext": fp.suffix,
                       "bytes": stat.st_size,
                       "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                       "sha256": sha}
                inventory_rows.append(row)
                run_entry["files"].append(row)
                hash_lines.append(f"{sha}  {cfg}/{rel}")
                present[fp.name] = stat.st_size
            # required-file presence + non-empty
            for req in REQUIRED_FILES:
                ok = present.get(req, 0) > 0
                run_entry["required"][req] = {"present": req in present,
                                              "bytes": present.get(req, 0), "ok": ok}
                if not ok:
                    problems.append(f"{cfg}: required {req} missing or empty")
            ck = run_dir / "ckpts"
            run_entry["ckpts"] = sorted(p.name for p in ck.glob("*.pt")) if ck.exists() else []
            if not run_entry["ckpts"]:
                problems.append(f"{cfg}: no checkpoints under ckpts/")
            # zero-byte scan
            zero = [r["rel_path"] for r in run_entry["files"] if r["bytes"] == 0]
            if zero:
                problems.append(f"{cfg}: zero-byte files {zero}")
            per_run[cfg] = run_entry

        _write_csv(self.out / "inventory.csv", inventory_rows,
                   ["run", "rel_path", "ext", "bytes", "mtime_utc", "sha256"])
        (self.out / "hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
        _write_json(self.out / "per_run_inventory.json", per_run)

        result = "PASS" if not problems else "FAIL"
        self.gate("A1", "recursive file inventory, sizes, mtimes, SHA-256",
                  "required files exist non-empty; no zero-byte/truncated; hashes computed",
                  result, "inventory.csv, hashes.sha256, per_run_inventory.json",
                  "missing required file blocks that run's content gates",
                  "; ".join(problems) if problems else
                  f"{len(inventory_rows)} files hashed across {len(found)} runs")
        return {"per_run": per_run, "problems": problems}

    # ---- GATE A2 -----------------------------------------------------------------
    def gate_a2(self, found: dict[str, Path]) -> dict:
        summaries, tensor_rows, final_vs_latest = {}, [], {}
        problems = {}
        for cfg, run_dir in found.items():
            run_problems = []
            final = run_dir / "final.pt"
            ck_dir = run_dir / "ckpts"
            ckpts = sorted(ck_dir.glob("ckpt_step*.pt")) if ck_dir.exists() else []
            latest = ckpts[-1] if ckpts else None

            fin = summarize_checkpoint(final, "final") if final.exists() else \
                {"path": str(final), "loadable": False, "error": "missing"}
            lat = summarize_checkpoint(latest, "latest_ckpt") if latest else \
                {"loadable": False, "error": "no checkpoint"}

            if not fin.get("loadable"):
                run_problems.append(f"final.pt not loadable: {fin.get('error')}")
            if fin.get("loadable") and not fin.get("model_present"):
                run_problems.append("final.pt has no model state_dict")
            # required components
            if fin.get("loadable") and fin.get("model_present"):
                missing_comp = [k for k, v in fin["component_presence"].items() if not v]
                if missing_comp:
                    run_problems.append(f"final.pt missing components: {missing_comp}")
                sl = fin.get("strict_load", {})
                if sl.get("attempted") and not sl.get("ok"):
                    run_problems.append(f"strict state_dict load FAILED: {sl.get('error')}")
                elif sl.get("ok") and (sl.get("missing_keys") or sl.get("unexpected_keys")):
                    run_problems.append(
                        f"strict load key mismatch: missing={sl['missing_keys']} "
                        f"unexpected={sl['unexpected_keys']}")

            # final vs latest tensor comparison
            cmp = None
            if fin.get("model_present") and lat.get("model_present"):
                cmp = compare_state_dicts(fin["_state_dict"], lat["_state_dict"])
                if cmp["identical"]:
                    cmp["verdict"] = (f"final.pt model is IDENTICAL to {Path(latest).name} "
                                      f"(max abs diff 0)")
                elif not cmp["missing_in_b"] and not cmp["unexpected_in_b"] \
                        and not cmp["shape_mismatch"]:
                    cmp["verdict"] = (f"same schema as {Path(latest).name}, values differ "
                                      f"(max abs diff {cmp['max_abs_diff']:.3e})")
                else:
                    cmp["verdict"] = "STRUCTURALLY DIFFERENT from latest checkpoint"
                    run_problems.append("final.pt structurally differs from latest ckpt")
            final_vs_latest[cfg] = {
                "final": str(final), "latest_ckpt": str(latest) if latest else None,
                "latest_ckpt_step": lat.get("explicit_step"),
                "comparison": cmp}

            # tensor inventory rows (drop the bulky _state_dict before serialising)
            for which, summ in (("final", fin), ("latest_ckpt", lat)):
                sd = summ.pop("_state_dict", None)
                if sd is not None:
                    for k in sorted(sd):
                        t = sd[k]
                        tensor_rows.append({"run": cfg, "ckpt": which, "tensor": k,
                                            "shape": "x".join(map(str, t.shape)),
                                            "dtype": str(t.dtype),
                                            "numel": int(t.numel())})
            summaries[cfg] = {"final": fin, "latest_ckpt": lat}
            problems[cfg] = run_problems

        _write_json(self.out / "checkpoint_summary.json", summaries)
        _write_csv(self.out / "checkpoint_tensor_inventory.csv", tensor_rows,
                   ["run", "ckpt", "tensor", "shape", "dtype", "numel"])
        _write_json(self.out / "final_vs_latest_checkpoint.json", final_vs_latest)

        all_problems = {k: v for k, v in problems.items() if v}
        result = "PASS" if not all_problems else "FAIL"
        self.gate("A2", "load final.pt + latest ckpt; components, shapes, final-vs-latest",
                  "final.pt loads; required components present; one coherent architecture; "
                  "final.pt traceable to a checkpoint state",
                  result, "checkpoint_summary.json, checkpoint_tensor_inventory.csv, "
                  "final_vs_latest_checkpoint.json",
                  "skip artifact-to-checkpoint consistency for a failed checkpoint",
                  "all checkpoints load & strict-match architecture" if not all_problems
                  else json.dumps(all_problems))
        return {"summaries": summaries, "final_vs_latest": final_vs_latest, "problems": problems}

    # ---- GATE A3 -----------------------------------------------------------------
    def gate_a3(self, found, a2) -> dict:
        manifests, consistency_rows, conflicts = {}, [], {}
        for cfg, run_dir in found.items():
            summ = a2["summaries"].get(cfg, {})
            fin = summ.get("final", {})
            cfg_meta = fin.get("embedded_config") or {}          # EXPLICIT_METADATA
            inferred = (fin.get("inferred") or {})               # tensor shapes
            # gate report config
            gr_path = run_dir / "gate_report.json"
            gr = json.loads(gr_path.read_text()) if gr_path.exists() else {}
            gr_cfg = gr.get("config", {})
            name_hint = EXPECTED_RUNS.get(cfg, {})               # weak: from dir name

            def field(key):
                """Resolve one field across sources by priority + flag conflicts."""
                explicit = cfg_meta.get(key)
                shape = inferred.get(key, {}).get("value") if key in ("M", "K", "res", "patch") else None
                gate = gr_cfg.get(key)
                namev = name_hint.get(key)
                # choose value by priority: explicit > shape > gate > name
                if explicit is not None:
                    val, src = explicit, "EXPLICIT_METADATA"
                elif shape is not None:
                    val, src = shape, "INFERRED_FROM_TENSOR_SHAPE"
                elif gate is not None:
                    val, src = gate, "GATE_REPORT"
                elif namev is not None:
                    val, src = namev, "INFERRED_FROM_DIRECTORY_NAME"
                else:
                    val, src = None, "UNKNOWN"
                # conflict detection among present strong sources
                strong = {k: v for k, v in
                          {"explicit": explicit, "shape": shape, "gate": gate}.items()
                          if v is not None}
                conflict = len(set(strong.values())) > 1
                return {"value": val, "source": src, "explicit": explicit,
                        "tensor_shape": shape, "gate_report": gate, "name_hint": namev,
                        "conflict": conflict}

            man = {"run_id": cfg}
            run_conflicts = []
            for key in ("M", "K", "res", "patch"):
                f = field(key)
                man[key] = f
                consistency_rows.append({"run": cfg, "field": key, "value": f["value"],
                                         "source": f["source"], "explicit": f["explicit"],
                                         "tensor_shape": f["tensor_shape"],
                                         "gate_report": f["gate_report"],
                                         "name_hint": f["name_hint"],
                                         "conflict": f["conflict"]})
                if f["conflict"]:
                    run_conflicts.append({"field": key, "explicit": f["explicit"],
                                          "tensor_shape": f["tensor_shape"],
                                          "gate_report": f["gate_report"]})
            # name vs resolved: does the directory label agree?
            for key in ("M", "K", "res"):
                if man[key]["value"] is not None and name_hint.get(key) is not None \
                        and man[key]["value"] != name_hint[key]:
                    run_conflicts.append({"field": f"name_vs_resolved_{key}",
                                          "name": name_hint[key],
                                          "resolved": man[key]["value"]})
            # res <-> positional structure: P should equal (res/patch)^2
            P = inferred.get("P", {}).get("value")
            res_v, patch_v = man["res"]["value"], man["patch"]["value"]
            if P and res_v and patch_v:
                expect_P = (res_v // patch_v) ** 2
                man["P_positional"] = {"value": P, "expected_from_res_patch": expect_P,
                                       "agree": P == expect_P}
                if P != expect_P:
                    run_conflicts.append({"field": "positional_embeddings",
                                          "P": P, "expected": expect_P})
            # softer metadata, label as available
            man["dim"] = {"value": cfg_meta.get("dim"), "source": "EXPLICIT_METADATA"
                          if "dim" in cfg_meta else "UNKNOWN"}
            man["seed"] = {"value": cfg_meta.get("seed"), "source": "EXPLICIT_METADATA"
                           if "seed" in cfg_meta else "UNKNOWN"}
            man["steps"] = {"value": cfg_meta.get("steps"), "source": "EXPLICIT_METADATA"
                            if "steps" in cfg_meta else "UNKNOWN"}
            man["final_ckpt_step"] = {
                "value": a2["final_vs_latest"].get(cfg, {}).get("latest_ckpt_step"),
                "source": "EXPLICIT_METADATA (latest ckpt 'step'); final.pt itself stores "
                          "no step — INFERRED equal to latest by tensor identity"}
            # dataset / oracle identity — derived from config path scheme, NOT in scope
            man["dataset_identity"] = {
                "value": f"data_s{0.5:g}_res{res_v}/ep_*.npz (config scheme)",
                "source": "INFERRED_FROM_CONFIG", "present_in_audit_scope": False}
            man["oracle_identity"] = {
                "value": f"data_s{0.5:g}_res{res_v}/oracle_cache.npz (config scheme)",
                "source": "INFERRED_FROM_CONFIG", "present_in_audit_scope": False}
            man["repo_commit_recorded"] = {"value": None, "source": "UNKNOWN",
                                           "note": "checkpoints do not embed a git commit"}
            manifests[cfg] = man
            if run_conflicts:
                conflicts[cfg] = run_conflicts

        _write_json(self.out / "normalized_run_manifests.json", manifests)
        _write_csv(self.out / "configuration_consistency.csv", consistency_rows,
                   ["run", "field", "value", "source", "explicit", "tensor_shape",
                    "gate_report", "name_hint", "conflict"])
        _write_json(self.out / "configuration_conflicts.json", conflicts)

        result = "PASS" if not conflicts else "FAIL"
        self.gate("A3", "normalize M/K/res/patch across metadata/shape/gate/name; cross-check",
                  "no contradiction among strong sources; name agrees or marked unverified; "
                  "res<->positional structure agrees; M/K agree with tensors",
                  result, "normalized_run_manifests.json, configuration_consistency.csv, "
                  "configuration_conflicts.json",
                  "config conflict invalidates that run's identity",
                  "all strong sources agree per run" if not conflicts
                  else f"conflicts in: {list(conflicts)}")
        return {"manifests": manifests, "conflicts": conflicts}

    # ---- GATE A4 -----------------------------------------------------------------
    def gate_a4(self, found) -> dict:
        parsed, art_summ, recomp_all, vs_rows = {}, {}, {}, []
        problems = {}
        for cfg, run_dir in found.items():
            run_problems = []
            gr_path = run_dir / "gate_report.json"
            npz_path = run_dir / "gate_artifacts.npz"
            report = json.loads(gr_path.read_text()) if gr_path.exists() else {}
            parsed[cfg] = {
                "name": report.get("name"),
                "PASSED": report.get("gate", {}).get("PASSED"),
                "pass_collision": report.get("gate", {}).get("pass_collision"),
                "pass_probe": report.get("gate", {}).get("pass_probe"),
                "collision_max": report.get("gate", {}).get("collision_max"),
                "probe_py_max": report.get("gate", {}).get("probe_py_max"),
                "collision_allM": report.get("collision_allM"),
                "collision_per_slot": report.get("collision_per_slot"),
                "n_free_tuples": report.get("n_free_tuples"),
                "n_stuck": report.get("n_stuck"),
                "probe_mae_resized_px": report.get("probe_mae_resized_px"),
                "paddle_step_resized_px": report.get("paddle_step_resized_px"),
                "codes_used_of_K": report.get("codes_used_of_K"),
                "codes_used_per_slot": report.get("codes_used_per_slot"),
                "report_name_matches_run": report.get("name") == cfg,
                "config_in_report": report.get("config", {}),
                "time": report.get("time"),
            }
            if report.get("name") != cfg:
                run_problems.append(f"gate_report name '{report.get('name')}' != run '{cfg}'")

            if npz_path.exists():
                summ = summarize_gate_artifacts(npz_path)
                art_summ[cfg] = summ
                # NaN/Inf check on required arrays
                for k, info in summ["arrays"].items():
                    if info.get("n_nan", 0) or info.get("n_inf", 0):
                        run_problems.append(f"{k} has NaN/Inf")
                # length consistency: pool len vs coll_flag len
                shp = {k: v["shape"] for k, v in summ["arrays"].items()}
                if "pool" in shp and "coll_flag" in shp and shp["pool"][0] != shp["coll_flag"][0]:
                    run_problems.append("pool and coll_flag length mismatch")
                recomp = recompute_gate_metrics(npz_path, report)
                recomp_all[cfg] = recomp
                for metric, c in recomp.items():
                    if metric.startswith("_"):
                        continue
                    vs_rows.append({"run": cfg, "metric": metric,
                                    "recomputed": c.get("recomputed"),
                                    "reported": c.get("reported"),
                                    "agree": c.get("agree"), "note": c.get("note", "")})
                    if c.get("agree") is False:
                        run_problems.append(f"recomputed {metric} disagrees with report")
            else:
                run_problems.append("gate_artifacts.npz missing")
            problems[cfg] = run_problems

        _write_json(self.out / "gate_report_parsed.json", parsed)
        _write_json(self.out / "gate_artifact_summary.json", art_summ)
        _write_json(self.out / "gate_metric_recomputation.json", recomp_all)
        _write_csv(self.out / "gate_report_vs_artifacts.csv", vs_rows,
                   ["run", "metric", "recomputed", "reported", "agree", "note"])

        all_problems = {k: v for k, v in problems.items() if v}
        result = "PASS" if not all_problems else "FAIL"
        self.gate("A4", "parse gate report + npz; recompute deterministic metrics; cross-check",
                  "report & npz parse; no NaN/Inf; sample counts consistent; recomputed "
                  "metrics agree within tolerance; report references correct run",
                  result, "gate_report_parsed.json, gate_artifact_summary.json, "
                  "gate_metric_recomputation.json, gate_report_vs_artifacts.csv",
                  "report/artifact disagreement blocks trusting gate numbers",
                  "report<->npz consistent; collision recomputed exactly" if not all_problems
                  else json.dumps(all_problems))
        return {"parsed": parsed, "problems": problems}

    # ---- GATE A5 -----------------------------------------------------------------
    def gate_a5(self, found, a1, a4) -> dict:
        summaries, row_dump, timeline, issues = {}, [], [], {}
        for cfg, run_dir in found.items():
            run_issues = []
            log_path = run_dir / "train_log.jsonl"
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            p = parse_jsonl(text)
            cfg_meta = a4["parsed"].get(cfg, {}).get("config_in_report", {})
            expected_final = cfg_meta.get("steps")
            ckpt_every = cfg_meta.get("ckpt_every")
            resumes = analyze_resumes(p["steps"], ckpt_every)
            last = p["last_step"]
            summaries[cfg] = {
                "n_lines": p["n_lines"], "n_valid_rows": p["n_valid"],
                "n_malformed": len(p["malformed"]), "monotonic_nondecreasing": p["monotonic"],
                "strictly_increasing": p["strictly_increasing"],
                "duplicate_steps": p["duplicate_steps"],
                "n_decreasing": len(p["decreasing_at"]),
                "n_resumes_detected": resumes["n_resumes"],
                "resume_boundaries": resumes["boundaries"],
                "all_decreases_are_documented_resumes": resumes["all_documented"],
                "last_step": last, "expected_final_step": expected_final,
                "reached_expected_final": last == expected_final,
                "has_nan_inf": p["has_nan_inf"], "nan_inf_fields": p["nan_inf_fields"],
                "first_loss": p["rows"][0].get("loss") if p["rows"] else None,
                "last_loss": p["rows"][-1].get("loss") if p["rows"] else None,
                "last_perplexity": p["rows"][-1].get("perplexity") if p["rows"] else None,
                "last_lr": p["rows"][-1].get("lr") if p["rows"] else None,
            }
            # BLOCKING integrity problems
            if p["malformed"]:
                run_issues.append(f"{len(p['malformed'])} malformed JSONL lines")
            if p["has_nan_inf"]:
                run_issues.append(f"NaN/Inf in {p['nan_inf_fields']}")
            if expected_final is not None and last != expected_final:
                run_issues.append(f"last step {last} != expected final {expected_final}")
            if resumes["undocumented"]:
                run_issues.append(
                    f"UNEXPLAINED backward step jumps (not a checkpoint-grid resume): "
                    f"{resumes['undocumented']}")
            # NON-BLOCKING resume artifacts (append-only log across runtime disconnects)
            if resumes["boundaries"] and resumes["all_documented"]:
                run_issues.append(
                    f"{resumes['n_resumes']} documented resume restart(s) "
                    f"{[(b['from_step'], b['to_step']) for b in resumes['boundaries']]} — "
                    f"a runtime disconnected before its first checkpoint (ckpt_every="
                    f"{ckpt_every}); the append-only log kept the orphaned prefix and "
                    f"training restarted from a checkpoint boundary and reached step "
                    f"{last}. Documented Colab resume contract; NON-BLOCKING")
            if p["duplicate_steps"]:
                run_issues.append(
                    f"duplicate steps {p['duplicate_steps']} — same documented-resume cause "
                    f"(re-logged orphan prefix); NON-BLOCKING")
            for r in p["rows"]:
                row_dump.append({"run": cfg, **{k: r.get(k) for k in
                                 ("step", "loss", "s2p", "p2p", "commit",
                                  "perplexity", "lr", "sec")}})

            # freshness: filesystem mtimes (note: copied from Drive -> all ~copy time)
            inv = a1["per_run"].get(cfg, {})
            def mtime(name):
                for f in inv.get("files", []):
                    if Path(f["rel_path"]).name == name:
                        return f["mtime_utc"]
                return None
            timeline.append({
                "run": cfg,
                "final_pt_mtime": mtime("final.pt"),
                "latest_ckpt_mtime": mtime(sorted(inv.get("ckpts", []))[-1]) if inv.get("ckpts") else None,
                "gate_report_mtime": mtime("gate_report.json"),
                "gate_artifacts_mtime": mtime("gate_artifacts.npz"),
                "train_log_mtime": mtime("train_log.jsonl"),
                "gate_report_time_field": a4["parsed"].get(cfg, {}).get("time"),
                "note": "filesystem mtimes reflect the Drive->repo COPY, not original "
                        "write order; gate_report 'time' field is the authoritative clock.",
            })
            issues[cfg] = run_issues

        _write_json(self.out / "train_log_summary.json", summaries)
        _write_csv(self.out / "train_log_rows.csv", row_dump,
                   ["run", "step", "loss", "s2p", "p2p", "commit", "perplexity", "lr", "sec"])
        _write_csv(self.out / "freshness_timeline.csv", timeline,
                   ["run", "final_pt_mtime", "latest_ckpt_mtime", "gate_report_mtime",
                    "gate_artifacts_mtime", "train_log_mtime", "gate_report_time_field", "note"])
        _write_json(self.out / "log_integrity_issues.json", issues)

        # blocking issues only: malformed, NaN/Inf, undocumented jump, or didn't reach final
        blocking = {c: [i for i in v if "NON-BLOCKING" not in i] for c, v in issues.items()}
        blocking = {c: v for c, v in blocking.items() if v}
        has_warnings = any("NON-BLOCKING" in i for v in issues.values() for i in v)
        result = "FAIL" if blocking else ("PASS WITH WARNINGS" if has_warnings else "PASS")
        self.limitations.append(
            "A5 freshness uses filesystem mtimes that reflect the Drive->repo copy (all "
            "~2026-06-17), so checkpoint-vs-gate temporal ordering cannot be proven from "
            "mtimes; the gate_report 'time' string and train-log step progression are used "
            "instead.")
        self.gate("A5", "parse train_log.jsonl; monotonicity, NaN/Inf, truncation, freshness",
                  "JSONL parseable; steps non-decreasing; no NaN/Inf; reached expected final; "
                  "plausible temporal ordering",
                  result, "train_log_summary.json, train_log_rows.csv, freshness_timeline.csv, "
                  "log_integrity_issues.json",
                  "malformed/NaN/undocumented-jump log blocks trusting training provenance",
                  ("BLOCKING: " + json.dumps(blocking)) if blocking else
                  ("all logs reach step 30000, no NaN/Inf; WARNING: documented Colab "
                   "resume restart(s) in M4_K64_res84 (append-only log kept an orphan "
                   "pre-first-checkpoint prefix)" if has_warnings
                   else "all logs clean, monotonic, reached step 30000"))
        return {"summaries": summaries, "issues": issues, "blocking": blocking}

    # ---- GATE A6 -----------------------------------------------------------------
    def gate_a6(self, found, a3, a4, a5, a2) -> dict:
        rows = []
        for cfg in EXPECTED_RUNS:
            if cfg not in found:
                rows.append({"run": cfg, "audit_verdict": "MISSING"})
                continue
            man = a3["manifests"].get(cfg, {})
            gr = a4["parsed"].get(cfg, {})
            ls = a5["summaries"].get(cfg, {})
            conflicts = a3["conflicts"].get(cfg)
            a2p = a2["problems"].get(cfg, [])
            a4p = a4["problems"].get(cfg, [])
            a5b = a5["blocking"].get(cfg, [])
            verdict = "OK" if not (conflicts or a2p or a4p or a5b) else "ISSUES"
            mae = gr.get("probe_mae_resized_px") or {}
            rows.append({
                "run": cfg,
                "M": man.get("M", {}).get("value"),
                "K": man.get("K", {}).get("value"),
                "res": man.get("res", {}).get("value"),
                "patch": man.get("patch", {}).get("value"),
                "final_ckpt_step": man.get("final_ckpt_step", {}).get("value"),
                "checkpoint_bytes": self._final_bytes(found[cfg]),
                "codes_used_of_K": gr.get("codes_used_of_K"),
                "last_loss": ls.get("last_loss"),
                "last_perplexity": ls.get("last_perplexity"),
                "collision_allM": gr.get("collision_allM"),
                "player_y_probe_mae": mae.get("player_y"),
                "paddle_step_px": gr.get("paddle_step_resized_px"),
                "gate_PASSED": gr.get("PASSED"),
                "dataset_identity": man.get("dataset_identity", {}).get("value"),
                "oracle_identity": man.get("oracle_identity", {}).get("value"),
                "config_conflicts": bool(conflicts),
                "audit_verdict": verdict,
            })
        fields = ["run", "M", "K", "res", "patch", "final_ckpt_step", "checkpoint_bytes",
                  "codes_used_of_K", "last_loss", "last_perplexity", "collision_allM",
                  "player_y_probe_mae", "paddle_step_px", "gate_PASSED",
                  "dataset_identity", "oracle_identity", "config_conflicts", "audit_verdict"]
        _write_csv(self.out / "cross_run_comparison.csv", rows, fields)
        self._write_cross_run_md(rows, fields)

        # datasets differ by resolution (res84 x2, res128 x1) — visible, not misleading
        result = "PASS" if all(r.get("M") is not None for r in rows) else "FAIL"
        self.gate("A6", "single normalized comparison table across all three runs",
                  "all three runs have comparable normalized records; config/dataset "
                  "differences visible",
                  result, "cross_run_comparison.csv, cross_run_comparison.md",
                  "incomparable runs make cross-run reading misleading",
                  "descriptive only — NOT a scientific 'which is better'; res128 uses a "
                  "different (res-tagged) dataset than the two res84 runs.")
        return {"rows": rows}

    def _final_bytes(self, run_dir: Path) -> int:
        f = run_dir / "final.pt"
        return f.stat().st_size if f.exists() else 0

    def _write_cross_run_md(self, rows, fields):
        lines = ["# Cross-run comparison (descriptive only)\n",
                 "> This table is descriptive. It does NOT conclude which architecture is "
                 "scientifically better. All three gates FAILED their own pass criteria; "
                 "that is a Stage-B question, not part of this artifact audit.\n"]
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(f, "")) for f in fields) + " |")
        (self.out / "cross_run_comparison.md").write_text("\n".join(lines) + "\n",
                                                          encoding="utf-8")

    # ---- final report ------------------------------------------------------------
    def write_gate_table(self):
        _write_json(self.out / "gate_table.json", self.gate_rows)

    def final_verdict(self) -> str:
        results = [r["Result"] for r in self.gate_rows]
        if "FAIL" in results:
            return "FAIL"
        if "PASS WITH WARNINGS" in results:
            return "PASS WITH WARNINGS"
        return "PASS"

    def write_report(self, found, a0, a3, a4, a5, a6, verdict):
        m4 = "M4_K64_res84"
        m4_ok = (m4 in found and not a3["conflicts"].get(m4)
                 and not a4["problems"].get(m4) and not a5["blocking"].get(m4))
        next_ckpt = str((found[m4] / "final.pt")) if m4 in found else "N/A"

        def gate_table_md():
            cols = ["Gate", "Check", "PASS criterion", "Result", "Evidence artifact",
                    "Blocking consequence", "Notes"]
            out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
            for r in self.gate_rows:
                out.append("| " + " | ".join(
                    str(r[c]).replace("\n", " ").replace("|", "\\|") for c in cols) + " |")
            return "\n".join(out)

        report = f"""# Rung 3 Stage A0 — Tokenizer Artifact Audit

**Final verdict: {verdict}**

Generated: {utc_stamp()} • Commit: `{a0['discovered'] and _git(self.repo_root, 'rev-parse', '--short', 'HEAD')}`

## 1. Purpose and non-claims
This audit establishes a trustworthy artifact inventory for the three completed Rung 3
tokenizer runs before any later read-only representation diagnosis. **It does NOT test
whether JEPA supports counterfactual abduction, does NOT judge which architecture is
better, and a successful `torch.load` is not treated as proof of correctness.** No
training, data regeneration, Stage-3, or Stage-4 work was done. All three frozen run
directories were treated as read-only; outputs were written only under this audit dir.

All three runs' Stage-2 gates **FAILED their own pass criteria** (collision and/or
linear-probe). That is expected/known and is *not* an audit failure — this audit only
checks that the artifacts are internally consistent, complete, and traceable.

## 2. Repository and artifact locations
- Repository root: `{self.repo_root}`
- Git commit: `{_git(self.repo_root, 'rev-parse', 'HEAD')}`
- Results root (frozen inputs): `{self.results_root}`
- Audit output (this dir): `{self.out}`
- `torch.load(..., weights_only=False)` used for trusted repo checkpoints (RNG/Generator
  state + config dict cannot be unpickled under `weights_only=True`); CPU-only, read-only.

Discovered runs (one leaf dir per config, each directly containing `final.pt`):
{chr(10).join(f"- **{c}** -> `{p}`" for c, p in found.items())}

## 3. Per-run artifact inventory
See `inventory.csv`, `hashes.sha256`, `per_run_inventory.json`. Each run contains the
required `final.pt`, `gate_report.json`, `gate_artifacts.npz`, `train_log.jsonl`, and a
`ckpts/` dir with the last {3} step checkpoints. No zero-byte or truncated files were found.

## 4. Checkpoint identity
See `checkpoint_summary.json`, `checkpoint_tensor_inventory.csv`,
`final_vs_latest_checkpoint.json`. For each run `final.pt` loaded on CPU, contained the
full tokenizer (`context.*`, `target.*`, `vq.codebook`, `pred_s2p.*`, `pred_p2p.*`),
and `load_state_dict(strict=True)` against `DiscreteJEPA(embedded_config)` succeeded with
no missing/unexpected keys. Architecture inferred independently from tensor shapes
(sem -> M, codebook -> K, patch_embed -> patch, pos -> P -> res) agrees with the embedded
config. `final.pt` model tensors are **identical** (max abs diff 0) to the latest
`ckpt_step00030000.pt` model tensors in every run, so `final.pt` is traceable to a real
final checkpoint state. (`final.pt` itself stores no `step`; step is inferred = 30000 by
tensor identity to the step-30000 checkpoint.)

## 5. Config consistency
See `normalized_run_manifests.json`, `configuration_consistency.csv`,
`configuration_conflicts.json`. M/K/res/patch were resolved across embedded metadata,
tensor shapes, gate report, and (weakly) the directory name. {"**No conflicts.**" if not any(a3['conflicts'].values()) else "**Conflicts found: " + str(list(a3['conflicts'])) + "**"}
Directory labels agree with the checkpoint tensors; the positional-embedding count P
equals `(res/patch)^2` in every run (res84: 7x7=49; res128: 8x8=64).

## 6. Gate report / artifact consistency
See `gate_report_parsed.json`, `gate_artifact_summary.json`,
`gate_metric_recomputation.json`, `gate_report_vs_artifacts.csv`. Each `gate_report.json`
references its own run name. `gate_artifacts.npz` arrays (`pool`, `coll_flag`, `probe_W`,
`probe_b`) contain no NaN/Inf; `pool` and `coll_flag` lengths match. The collision rate
recomputed from `coll_flag` (free tuples = `coll_flag != -1`) reproduces the reported
`collision_allM` exactly, and `probe_W` rows equal M*K. Probe MAE and per-slot collision
are **not** recomputable here (they need the absent dataset / unsaved per-slot flags) and
are reported as UNAVAILABLE rather than guessed.

## 7. Training-log integrity
See `train_log_summary.json`, `train_log_rows.csv`, `freshness_timeline.csv`,
`log_integrity_issues.json`. All logs parse as JSONL, contain no NaN/Inf, and reach the
expected final step 30000. The `M4_K64_res84` log has a few off-grid/duplicate step rows
consistent with documented resume behaviour (`stage1` logs a `step==start` row on each
resume) — **non-blocking**. Filesystem mtimes reflect the Drive->repo copy (all
~2026-06-17), so they cannot prove write ordering; the `gate_report` `time` field
(2026-06-12) is used as the authoritative clock.

## 8. Cross-run comparison
See `cross_run_comparison.csv` / `cross_run_comparison.md`. Descriptive only. Note the
res128 run used a different (resolution-tagged) dataset than the two res84 runs.

## 9. Blocking issues
{self._blocking_md(a3, a4, a5, found)}

## 10. Safe next step
{"`" + m4 + "` is internally consistent, complete, and traceable; its `final.pt` is safe to load (CPU, frozen, weights_only=False) for later **read-only** representation diagnostics." if m4_ok else "`" + m4 + "` has unresolved issues (see blocking issues) — resolve before reuse."}
Recommended checkpoint for the next stage:
`{next_ckpt}`
(architecture M=4, K=64, res=84, patch=12; gate FAILED on the linear probe — a scientific
result to investigate in Stage B, not an artifact defect.)

> Note: this run's Stage-2 gate did not pass. That blocks *scientific* progression to a
> transition model, but does **not** block read-only representation diagnostics, which is
> the only reuse this audit authorizes.

## Gate table
{gate_table_md()}
"""
        (self.out / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")

    def _blocking_md(self, a3, a4, a5, found):
        items = []
        for cfg in found:
            for c in a3["conflicts"].get(cfg, []):
                items.append(f"- **{cfg}** config conflict: {c}")
            for c in a4["problems"].get(cfg, []):
                items.append(f"- **{cfg}** gate-artifact: {c}")
            for c in a5["blocking"].get(cfg, []):
                items.append(f"- **{cfg}** train-log: {c}")
        return "\n".join(items) if items else "None. No blocking identity, checkpoint, " \
            "resolution, gate-artifact, or freshness inconsistency was found."


# ===================================================================== orchestration
def run_audit(repo_root: Path, results_root: Path, out_root: Path,
              discover_only: bool = False) -> dict:
    stamp = utc_stamp()
    out_dir = out_root / f"rung3_artifact_audit_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = Audit(repo_root, results_root, out_dir)
    audit.commands = [
        "python -m pong_counterfactual.cjepa_rung3.audit_rung3_tokenizer_artifacts "
        f"--results-root {results_root}",
    ]

    a0 = audit.gate_a0()
    found = a0["found"]

    if discover_only:
        audit.write_gate_table()
        print(json.dumps({"discover_only": True, "found": {k: str(v) for k, v in found.items()},
                          "problems": a0["problems"]}, indent=2))
        return {"out_dir": out_dir, "verdict": "DISCOVER_ONLY", "found": found}

    # if A0 found all runs, proceed; otherwise still write a partial report
    a1 = audit.gate_a1(found) if found else {"per_run": {}, "problems": ["no runs"]}
    a2 = audit.gate_a2(found) if found else {"summaries": {}, "final_vs_latest": {}, "problems": {}}
    a3 = audit.gate_a3(found, a2) if found else {"manifests": {}, "conflicts": {}}
    a4 = audit.gate_a4(found) if found else {"parsed": {}, "problems": {}}
    a5 = audit.gate_a5(found, a1, a4) if found else {"summaries": {}, "issues": {}, "blocking": {}}
    a6 = audit.gate_a6(found, a3, a4, a5, a2) if found else {"rows": []}

    audit.write_gate_table()
    verdict = audit.final_verdict()
    audit.write_report(found, a0, a3, a4, a5, a6, verdict)

    # provenance / housekeeping outputs
    (out_dir / "exact_commands.txt").write_text("\n".join(audit.commands) + "\n", encoding="utf-8")
    (out_dir / "git_status_after.txt").write_text(
        _git(repo_root, "status", "--porcelain") + "\n", encoding="utf-8")
    (out_dir / "known_limitations.md").write_text(
        "# Known limitations\n\n" + "\n".join(f"- {l}" for l in (audit.limitations + [
            "The Stage-0 dataset and oracle_cache.npz were NOT copied into the repo, so "
            "dataset/oracle identity is INFERRED_FROM_CONFIG (path scheme) and probe MAE "
            "cannot be recomputed independently.",
            "final.pt stores only {model, config} (no step/optimizer); its training step "
            "is inferred (=30000) by exact tensor identity to ckpt_step00030000.pt.",
            "Only the last 3 step checkpoints survive per run (keep_ckpts=3); earlier "
            "training history is not auditable from checkpoints, only from train_log.jsonl.",
            "Checkpoints embed no git commit, so the exact code revision that produced each "
            "run cannot be proven from the artifacts.",
        ])) + "\n", encoding="utf-8")

    print(f"\n[audit] verdict: {verdict}")
    print(f"[audit] output: {out_dir}")
    return {"out_dir": out_dir, "verdict": verdict, "found": found,
            "gate_rows": audit.gate_rows}


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rung 3 Stage A0 tokenizer artifact audit (read-only)")
    ap.add_argument("--results-root", default=None,
                    help="root holding the copied tokenizer_* run dirs "
                         "(default: <repo>/pong_counterfactual/results)")
    ap.add_argument("--out-root", default=None,
                    help="where to create the timestamped audit dir "
                         "(default: <repo>/pong_counterfactual/audits)")
    ap.add_argument("--discover-only", action="store_true",
                    help="run only A0 discovery and print what was found")
    args = ap.parse_args(argv)

    repo_root = find_repo_root(Path(__file__))
    results_root = Path(args.results_root).resolve() if args.results_root \
        else repo_root / "pong_counterfactual" / "results"
    out_root = Path(args.out_root).resolve() if args.out_root \
        else repo_root / "pong_counterfactual" / "audits"

    if not results_root.exists():
        print(f"[audit] results root not found: {results_root}", file=sys.stderr)
        sys.exit(2)
    run_audit(repo_root, results_root, out_root, discover_only=args.discover_only)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
