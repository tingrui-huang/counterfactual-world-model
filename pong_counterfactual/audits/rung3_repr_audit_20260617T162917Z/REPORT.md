# Rung 3 Stage A1 — Representation Abductability Audit

**Final verdict: PASS WITH WARNINGS**

Generated: 20260617T162923Z • Checkpoint: `D:\Users\trhua\Research\counterfactual-world-model\pong_counterfactual\results\tokenizer_M4_K64_res84-20260617T152639Z-3-001\tokenizer_M4_K64_res84\final.pt` • s=0.5

## Purpose & non-claims
Tests whether the FROZEN `tokenizer_M4_K64_res84` representation carries the information a
later counterfactual abduction step would need. **A low collision rate alone is NOT taken
as proof of abductability** — the decisive test is whether the factual latent transition
z_t→z_{t+1} lets a linear probe decode sticky/executed-action info BEYOND the actions
themselves (sub-question C). No training, tokenizer modification, or Stage 3/4 was done.

## A. Branch validity
122/250 oracle
factual-vs-counterfactual branches differ physically (4-vector level)
(49%). Replay determinism: vec
1.000 /
frame 1.000.
Sticky=128, free=122, identifiable subset (a_prev≠a_int)
=250. Evidence: `sample_counts.json`, `data_integrity.json`,
`branch_physical_distance.csv`, `branch_distance_histogram.png`.

## B. Representation separability
Encoding deterministic: True. Measured on the
122 PHYSICALLY-DIFFERENT (free) pairs: token collision
**0.0%**, mean token Hamming
3.84/4, mean pre-quant latent dist
10.033. (Physically-identical/stuck pairs
collide 100% — correct, same frame; all-pairs
collision 51.2% merely equals the stuck rate and is NOT
the separability metric.) Cross-check vs the gate's saved coll_flag:
{'available': True, 'pool_match': True, 'n_free': 122, 'matches_saved_coll_flag': True}. Evidence:
`representation_metrics.json`, `pair_metrics.csv`, `token_hamming_histogram.png`,
`physical_vs_latent_distance.png`, `collision_by_physical_difference.png`, `pair_gallery.png`.

## C. Abductability
sticky[identifiable]: action-only 0.51 -> z_t+act 0.62 -> z_t+z_t1+act 0.57  (Δ_nextlatent=-0.05, Δ_vs_action=+0.06); executed_action[all]: action-only 0.51 -> z_t+act 0.62 -> z_t+z_t1+act 0.58  (Δ_nextlatent=-0.04, Δ_vs_action=+0.08) | next latent adds little/no info over z_t+actions and CIs touch chance — weak/uncertain signal | shuffled control near chance
Evidence: `probe_results.csv`, `sticky_probe_roc.png`, `sticky_probe_confusion_matrix.png`,
`executed_action_confusion_matrix.png`, `probe_summary.png`.

## Verdict rule applied
- PASS: latent transition clearly improves sticky/executed-action decoding over
  action-only, valid branches, control near chance.
- PASS WITH WARNINGS: some signal but weak confidence/coverage/performance.
- FAIL: invalid branches, representation collapse, or no usable latent-noise info.

**→ PASS WITH WARNINGS**

## Recommended next step
**improve temporal representation / compare with a reconstruction baseline before training a transition model**

## Gate table
| Gate | Check | PASS criterion | Result | Evidence artifact | Blocking consequence | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | data/oracle validity + branch physical difference | consistent lengths, 100% replay determinism, branches physically differ | PASS | sample_counts.json, data_integrity.json, branch_physical_distance.csv, branch_distance_histogram.png | if FAIL, R2/R3 are blocked (no trustworthy branches) | 122/250 branches differ physically |
| R2 | frozen encoding: determinism, collision, Hamming, latent dist | deterministic; physically-different branches stay token-distinct | PASS | representation_metrics.json, pair_metrics.csv, token_hamming_histogram.png, physical_vs_latent_distance.png, collision_by_physical_difference.png, pair_gallery.png | if FAIL, abductability (R3) cannot rest on these tokens | on 122 physically-DIFFERENT pairs: collision 0.0%, mean Hamming 3.84/4 (identical/stuck pairs collide 100%, as expected — same frame) |
| R3 | frozen-latent probes: does z_{t+1} add realized-noise info? | z_t+z_{t+1}+actions beats action-only on sticky/executed-action decode; shuffled control near chance | PASS WITH WARNINGS | probe_results.csv, sticky_probe_roc.png, sticky_probe_confusion_matrix.png, executed_action_confusion_matrix.png, probe_summary.png | drives the final abductability verdict | sticky[identifiable]: action-only 0.51 -> z_t+act 0.62 -> z_t+z_t1+act 0.57  (Δ_nextlatent=-0.05, Δ_vs_action=+0.06); executed_action[all]: action-only 0.51 -> z_t+act 0.62 -> z_t+z_t1+act 0.58  (Δ_nextlatent=-0.04, Δ_vs_action=+0.08) \| next latent adds little/no info over z_t+actions and CIs touch chance — weak/uncertain signal \| shuffled control near chance |
