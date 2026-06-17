# Known limitations

- A5 freshness uses filesystem mtimes that reflect the Drive->repo copy (all ~2026-06-17), so checkpoint-vs-gate temporal ordering cannot be proven from mtimes; the gate_report 'time' string and train-log step progression are used instead.
- The Stage-0 dataset and oracle_cache.npz were NOT copied into the repo, so dataset/oracle identity is INFERRED_FROM_CONFIG (path scheme) and probe MAE cannot be recomputed independently.
- final.pt stores only {model, config} (no step/optimizer); its training step is inferred (=30000) by exact tensor identity to ckpt_step00030000.pt.
- Only the last 3 step checkpoints survive per run (keep_ckpts=3); earlier training history is not auditable from checkpoints, only from train_log.jsonl.
- Checkpoints embed no git commit, so the exact code revision that produced each run cannot be proven from the artifacts.
