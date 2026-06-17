"""Minimal parser-correctness tests for the Rung 3 artifact audit utility.

These test the AUDIT TOOL's pure functions, not the research model. They use temporary
directories and tiny synthetic artifacts; they never touch the real frozen checkpoints.

Run:  python -m pytest pong_counterfactual/cjepa_rung3/test_audit_rung3_artifacts.py -q
"""
import hashlib

import numpy as np
import torch

from pong_counterfactual.cjepa_rung3 import audit_rung3_tokenizer_artifacts as A


# ----------------------------------------------------------------- JSONL parsing
def test_valid_jsonl_parses():
    text = '{"step": 50, "loss": 1.0}\n{"step": 100, "loss": 0.5}\n'
    r = A.parse_jsonl(text)
    assert r["n_valid"] == 2
    assert r["malformed"] == []
    assert r["monotonic"] and r["strictly_increasing"]
    assert r["last_step"] == 100
    assert not r["has_nan_inf"]


def test_malformed_jsonl_detected():
    text = '{"step": 50, "loss": 1.0}\n{this is not json}\n{"step": 100}\n'
    r = A.parse_jsonl(text)
    assert len(r["malformed"]) == 1
    assert r["malformed"][0]["line"] == 2
    assert r["n_valid"] == 2          # the two good lines still parse


def test_non_monotonic_steps_detected():
    text = '{"step": 100}\n{"step": 50}\n{"step": 60}\n'
    r = A.parse_jsonl(text)
    assert not r["monotonic"]
    assert r["decreasing_at"]         # at least one decrease recorded


def test_duplicate_steps_detected_but_nondecreasing():
    # resume semantics: a repeated step is a duplicate but not a decrease
    text = '{"step": 100}\n{"step": 100}\n{"step": 150}\n'
    r = A.parse_jsonl(text)
    assert r["duplicate_steps"] == [100]
    assert r["monotonic"]                 # non-decreasing
    assert not r["strictly_increasing"]   # but not strictly increasing


def test_nan_inf_detected():
    text = '{"step": 50, "loss": NaN}\n'
    # json.loads accepts NaN by default in CPython
    r = A.parse_jsonl(text)
    assert r["has_nan_inf"]
    assert "loss" in r["nan_inf_fields"]


# ----------------------------------------------------------------- resume analysis
def test_documented_resume_before_first_ckpt():
    # ran to 200, runtime died before first ckpt (500), restarted from 1, reached 30000
    steps = [1, 50, 100, 150, 200, 1, 50, 100, 150, 200, 250, 30000]
    r = A.analyze_resumes(steps, ckpt_every=500)
    assert r["n_resumes"] == 1
    assert r["all_documented"]                 # restart to 1 -> (1-1)%500==0
    assert r["undocumented"] == []


def test_documented_resume_from_checkpoint():
    # ran past ckpt 5000 to 5100, died, resumed from surviving ckpt 5000 -> logs 5001
    steps = [5000, 5050, 5100, 5001, 5050, 5100]
    r = A.analyze_resumes(steps, ckpt_every=500)
    assert r["n_resumes"] == 1                 # the 5100 -> 5001 drop
    assert r["all_documented"]                 # (5001-1)%500==0


def test_undocumented_backward_jump_flagged():
    # backward jump to a non-grid step -> not a documented resume
    steps = [50, 100, 7777, 50, 100]
    r = A.analyze_resumes(steps, ckpt_every=500)
    assert r["n_resumes"] == 1
    assert not r["all_documented"]
    assert r["undocumented"][0]["to_step"] == 50 or r["undocumented"]  # the 7777->50 drop


def test_no_resume_clean_log():
    r = A.analyze_resumes([50, 100, 150, 200], ckpt_every=500)
    assert r["n_resumes"] == 0 and r["all_documented"]


# ----------------------------------------------------------------- SHA-256 inventory
def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "blob.bin"
    data = b"rung3-audit-test-payload"
    f.write_bytes(data)
    assert A.sha256_file(f) == hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------- output-path guard
def test_output_inside_frozen_run_is_rejected(tmp_path):
    """A0 must FAIL when the audit output dir lands inside a discovered frozen run."""
    results = tmp_path / "results"
    run = results / "tokenizer_M4_K64_res84" / "tokenizer_M4_K64_res84"
    (run / "ckpts").mkdir(parents=True)
    (run / "final.pt").write_bytes(b"x")          # enough for discovery to classify it
    # output dir placed *inside* the frozen run
    bad_out = run / "audit_here"
    bad_out.mkdir()
    audit = A.Audit(repo_root=tmp_path, results_root=results, out_dir=bad_out)
    res = audit.gate_a0()
    assert any("frozen run" in p.lower() or "under results_root" in p.lower()
               for p in res["problems"])
    assert audit.gate_rows[-1]["Result"] == "FAIL"


def test_output_outside_frozen_run_passes_discovery(tmp_path):
    results = tmp_path / "results"
    for name in ("M4_K64_res84", "M8_K128_res84", "M4_K64_res128"):
        run = results / f"tokenizer_{name}" / f"tokenizer_{name}"
        (run / "ckpts").mkdir(parents=True)
        (run / "final.pt").write_bytes(b"x")
    good_out = tmp_path / "audits" / "run1"
    good_out.mkdir(parents=True)
    audit = A.Audit(repo_root=tmp_path, results_root=results, out_dir=good_out)
    res = audit.gate_a0()
    assert res["problems"] == []
    assert len(res["found"]) == 3
    assert audit.gate_rows[-1]["Result"] == "PASS"


# ----------------------------------------------------------------- state_dict compare
def _sd(**kw):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in kw.items()}


def test_compare_identical_state_dicts():
    a = _sd(w=[1.0, 2.0], b=[3.0])
    b = _sd(w=[1.0, 2.0], b=[3.0])
    r = A.compare_state_dicts(a, b)
    assert r["identical"]
    assert r["max_abs_diff"] == 0.0
    assert not r["missing_in_b"] and not r["unexpected_in_b"]


def test_compare_missing_key():
    a = _sd(w=[1.0], b=[2.0])
    b = _sd(w=[1.0])
    r = A.compare_state_dicts(a, b)
    assert r["missing_in_b"] == ["b"]
    assert not r["identical"]


def test_compare_shape_mismatch():
    a = _sd(w=[1.0, 2.0, 3.0])
    b = _sd(w=[1.0, 2.0])
    r = A.compare_state_dicts(a, b)
    assert r["shape_mismatch"] and r["shape_mismatch"][0]["key"] == "w"
    assert not r["identical"]


def test_compare_numerical_mismatch():
    a = _sd(w=[1.0, 2.0])
    b = _sd(w=[1.0, 2.5])
    r = A.compare_state_dicts(a, b)
    assert not r["identical"]
    assert abs(r["max_abs_diff"] - 0.5) < 1e-6
    assert not r["missing_in_b"] and not r["shape_mismatch"]


# ----------------------------------------------------------------- arch inference
def test_infer_arch_from_shapes():
    sd = {
        "context.sem": torch.zeros(1, 4, 128),
        "context.pos": torch.zeros(1, 49, 128),
        "context.patch_embed.weight": torch.zeros(128, 1, 12, 12),
        "vq.codebook": torch.zeros(64, 32),
    }
    inf = A.infer_arch_from_state_dict(sd)
    assert inf["M"]["value"] == 4
    assert inf["K"]["value"] == 64
    assert inf["patch"]["value"] == 12
    assert inf["P"]["value"] == 49
    assert inf["res"]["value"] == 84      # sqrt(49)*12
    assert inf["res"]["source"] == "INFERRED_FROM_TENSOR_SHAPE"


# ----------------------------------------------------------------- gate recompute
def test_recompute_collision_from_coll_flag(tmp_path):
    # 3 stuck (-1), 7 free of which 2 collide (1) -> collision_allM = 2/7
    coll = np.array([-1, -1, -1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64)
    pool = np.zeros((10, 2), dtype=np.int64)
    probe_W = np.zeros((4 * 64, 4), dtype=np.float32)   # M*K rows, 4 dims
    probe_b = np.zeros(4, dtype=np.float32)
    npz = tmp_path / "gate_artifacts.npz"
    np.savez(npz, pool=pool, coll_flag=coll, probe_W=probe_W, probe_b=probe_b)
    report = {"collision_allM": 2 / 7, "n_free_tuples": 7,
              "config": {"M": 4, "K": 64}}
    r = A.recompute_gate_metrics(npz, report)
    assert abs(r["collision_allM"]["recomputed"] - 2 / 7) < 1e-9
    assert r["collision_allM"]["agree"] is True
    assert r["n_free_tuples"]["recomputed"] == 7
    assert r["probe_W_MK"]["recomputed"] == 256
    assert r["probe_W_MK"]["agree"] is True
    assert r["probe_W_n_dims"]["agree"] is True
