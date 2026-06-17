# Rung 3 Stage A0 — Tokenizer Artifact Audit

**Final verdict: PASS WITH WARNINGS**

Generated: 20260617T155541Z • Commit: `4941a84`

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
- Repository root: `D:\Users\trhua\Research\counterfactual-world-model`
- Git commit: `4941a84390f48a761289f3db634e45ce1013bb0a`
- Results root (frozen inputs): `D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\results`
- Audit output (this dir): `D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\audits\rung3_artifact_audit_20260617T155536Z`
- `torch.load(..., weights_only=False)` used for trusted repo checkpoints (RNG/Generator
  state + config dict cannot be unpickled under `weights_only=True`); CPU-only, read-only.

Discovered runs (one leaf dir per config, each directly containing `final.pt`):
- **M4_K64_res84** -> `D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\results\tokenizer_M4_K64_res84-20260617T152639Z-3-001\tokenizer_M4_K64_res84`
- **M8_K128_res84** -> `D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\results\tokenizer_M8_K128_res84-20260617T152641Z-3-001\tokenizer_M8_K128_res84`
- **M4_K64_res128** -> `D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\results\tokenizer_M4_K64_res128-20260617T152640Z-3-001\tokenizer_M4_K64_res128`

## 3. Per-run artifact inventory
See `inventory.csv`, `hashes.sha256`, `per_run_inventory.json`. Each run contains the
required `final.pt`, `gate_report.json`, `gate_artifacts.npz`, `train_log.jsonl`, and a
`ckpts/` dir with the last 3 step checkpoints. No zero-byte or truncated files were found.

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
tensor shapes, gate report, and (weakly) the directory name. **No conflicts.**
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
None. No blocking identity, checkpoint, resolution, gate-artifact, or freshness inconsistency was found.

## 10. Safe next step
`M4_K64_res84` is internally consistent, complete, and traceable; its `final.pt` is safe to load (CPU, frozen, weights_only=False) for later **read-only** representation diagnostics.
Recommended checkpoint for the next stage:
`D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\results\tokenizer_M4_K64_res84-20260617T152639Z-3-001\tokenizer_M4_K64_res84\final.pt`
(architecture M=4, K=64, res=84, patch=12; gate FAILED on the linear probe — a scientific
result to investigate in Stage B, not an artifact defect.)

> Note: this run's Stage-2 gate did not pass. That blocks *scientific* progression to a
> transition model, but does **not** block read-only representation diagnostics, which is
> the only reuse this audit authorizes.

## Gate table
| Gate | Check | PASS criterion | Result | Evidence artifact | Blocking consequence | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | repo root + discover 3 runs + output-separation guard | exactly one dir per expected run; output outside frozen runs | PASS | repository_info.json, discovered_runs.json, git_status_before.txt | stop checkpoint-dependent gates for any run not uniquely identified | all 3 runs uniquely found |
| A1 | recursive file inventory, sizes, mtimes, SHA-256 | required files exist non-empty; no zero-byte/truncated; hashes computed | PASS | inventory.csv, hashes.sha256, per_run_inventory.json | missing required file blocks that run's content gates | 21 files hashed across 3 runs |
| A2 | load final.pt + latest ckpt; components, shapes, final-vs-latest | final.pt loads; required components present; one coherent architecture; final.pt traceable to a checkpoint state | PASS | checkpoint_summary.json, checkpoint_tensor_inventory.csv, final_vs_latest_checkpoint.json | skip artifact-to-checkpoint consistency for a failed checkpoint | all checkpoints load & strict-match architecture |
| A3 | normalize M/K/res/patch across metadata/shape/gate/name; cross-check | no contradiction among strong sources; name agrees or marked unverified; res<->positional structure agrees; M/K agree with tensors | PASS | normalized_run_manifests.json, configuration_consistency.csv, configuration_conflicts.json | config conflict invalidates that run's identity | all strong sources agree per run |
| A4 | parse gate report + npz; recompute deterministic metrics; cross-check | report & npz parse; no NaN/Inf; sample counts consistent; recomputed metrics agree within tolerance; report references correct run | PASS | gate_report_parsed.json, gate_artifact_summary.json, gate_metric_recomputation.json, gate_report_vs_artifacts.csv | report/artifact disagreement blocks trusting gate numbers | report<->npz consistent; collision recomputed exactly |
| A5 | parse train_log.jsonl; monotonicity, NaN/Inf, truncation, freshness | JSONL parseable; steps non-decreasing; no NaN/Inf; reached expected final; plausible temporal ordering | PASS WITH WARNINGS | train_log_summary.json, train_log_rows.csv, freshness_timeline.csv, log_integrity_issues.json | malformed/NaN/undocumented-jump log blocks trusting training provenance | all logs reach step 30000, no NaN/Inf; WARNING: documented Colab resume restart(s) in M4_K64_res84 (append-only log kept an orphan pre-first-checkpoint prefix) |
| A6 | single normalized comparison table across all three runs | all three runs have comparable normalized records; config/dataset differences visible | PASS | cross_run_comparison.csv, cross_run_comparison.md | incomparable runs make cross-run reading misleading | descriptive only — NOT a scientific 'which is better'; res128 uses a different (res-tagged) dataset than the two res84 runs. |
