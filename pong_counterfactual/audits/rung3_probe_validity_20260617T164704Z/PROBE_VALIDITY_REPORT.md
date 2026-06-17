# Rung 3 Stage A1.1 — Sticky-Noise Probe Validity Audit

**Verdict: PARTIALLY VALID**

Generated 20260617T164746Z • Checkpoint `final.pt` • read-only (no training, no Stage 3/4)

## TL;DR
The prior A1 R3 conclusion ("the JEPA latent transition carries no usable sticky /
executed-action information") **cannot be taken as a representation conclusion as stated**,
for three reasons this audit establishes:

1. **The binary `stuck` target is only weakly identifiable even from GROUND TRUTH.** Best
   oracle ROC-AUC = **0.659** (the full ground-truth 4-vector
   transition is ~chance; only |paddle displacement| is informative). The prior JEPA result
   (~0.67) already sits at this weak ceiling, so the representation cannot be blamed for the
   binary-stuck near-chance result.
2. **The old "identifiable" subset (a_prev ≠ a_intended) was VACUOUS** — a_prev is always
   NOOP here, so it selected ALL 250 tuples. The true oracle noise-identifiable subset is the
   122 FREE tuples; for the 128 stuck tuples the free counterfactual
   is replay-UNREACHABLE (o_fac==o_cf==other), so the binary stuck bit is not even
   counterfactually observable from one transition.
3. **z_t's apparent sticky-predictiveness is a finite-sample/overfit effect** (permutation
   p=0.154), not a genuine state→noise signal.

**But the prior's qualitative DIRECTION is corroborated by a proper test:** the
abduction-relevant *realized paddle move* (direction/displacement) is strongly present in
ground truth yet JEPA cannot decode it (best JEPA direction acc=0.582 vs
oracle 0.712 over base 0.401; displacement R²<0 even
from continuous pre-quant latents). So the eye genuinely fails to preserve the fine paddle
signal — consistent with its known coarse paddle localization — it was just measured against
the wrong (weakly-identifiable) target.

## 1. Feature encoding (feature_comparison.csv)
A1 R3 used slot-blocked **one-hot** tokens (M·K=256-d) — a VALID continuous encoding, NOT
raw integer IDs. Codebook embeddings and continuous **pre-quantization** latents were also
tested; none lifts sticky decoding above the weak oracle ceiling, so the conclusion is not
an artifact of quantization or feature coding.

## 2. Identifiability (identifiability_comparison.csv)
Old a_prev≠a_int = 250/250 (vacuous). Oracle noise-identifiable = 122
(free tuples). 128 stuck tuples have a replay-unreachable free branch.

## 3. Oracle upper bounds (oracle_upper_bounds.csv)
Binary stuck: best **0.659** (WEAK). Realized direction/displacement:
strongly recoverable from ground truth — the correct, strong ceiling.

## 4. Why z_t "predicted" sticky (state_label_bias.csv)
Permutation test: observed z_t-only AUC=0.551, null mean
0.500, p=0.154. Episode-grouped
splits used; per-episode stuck-rate spread and duplicate-state checks reported.

## 5. Corrected probes (grouped_split_results.csv)
Repeated grouped splits + bootstrap CIs, for binary stuck AND realized direction, across
encodings, with shuffled-label controls.

## 6. Collision reconciliation (collision_metric_reconciliation.md)
7.4% (gate, free, fac-vs-other, 1 step) / 51.2% (A1 all-pairs = stuck rate, trivial) /
0.0% (A1 fac-vs-CF on physically-different, 2 steps) — mutually consistent.

## Verdict: PARTIALLY VALID
The probe machinery (one-hot encoding, grouped splits) was largely sound, but the prior
result rested on a **vacuous identifiable subset, no oracle upper bound, a weakly-identifiable
target, and a non-significant z_t confound** — so it cannot, by itself, support a
representation conclusion. A corrected analysis against a strong upper bound (realized paddle
move) DOES show a genuine representation limitation, so the prior's recommendation
("improve the temporal / paddle representation") still stands — for the right reason.

## Recommended next step
Re-pose the abduction probe around the **realized paddle move (direction/displacement) on the
free-tuple identifiable subset**, with the oracle upper bound as the reference, before
deciding on architecture changes. Do NOT yet change the representation architecture.

## Gate table
| Gate | Check | PASS criterion | Result | Evidence artifact | Notes |
| --- | --- | --- | --- | --- | --- |
| C1 | token-ID feature encoding (one-hot/codebook/pre-quant vs raw IDs) | A1 used a VALID encoding; richer encodings tested for missed signal | PASS | feature_comparison.csv | A1 R3 used slot one-hot (VALID, not raw IDs). Best sticky AUC across all encodings = 0.579; no encoding (incl. continuous pre-quant) lifts it meaningfully above the weak oracle ceiling. |
| C2 | oracle noise-identifiable subset vs old a_prev!=a_int | identifiable defined by physical outcome change, not action labels | FAIL (old definition vacuous) | identifiability_comparison.csv | old 'identifiable'=250/250 was ALL tuples (a_prev always NOOP); true oracle-identifiable=122 (free tuples). For 128 stuck tuples the free counterfactual is replay-unreachable -> binary stuck is single-class on the identifiable set. |
| C3 | oracle upper bounds — is sticky recoverable from the transition at all? | establishes the ceiling before blaming the representation | WEAK CEILING | oracle_upper_bounds.csv | binary stuck: best oracle AUC=0.659 (full ground-truth 4-vec transition ~chance; only \|paddle disp\| informative) -> WEAK. Realized direction from ground truth acc=0.712 (base 0.40); displacement R2 from gt is high by construction. |
| C4 | why z_t appeared to predict sticky (bias/leakage/permutation) | z_t-only sticky AUC should be explained by confound or be non-significant | PASS | state_label_bias.csv | z_t-only sticky AUC permutation p=0.154; within null -> the prior z_t signal was finite-sample noise, not genuine. |
| C5 | corrected probes on oracle-aware targets, repeated grouped splits + CIs | z_t+z_t1+act vs action-only and vs oracle ceiling; control near chance | PASS WITH WARNINGS | grouped_split_results.csv | binary stuck: best JEPA transition AUC=0.579 (~oracle ceiling 0.66); realized direction: best JEPA acc=0.582 vs oracle 0.712 (base 0.401) -> JEPA cannot resolve the paddle move. |
| C6 | reconcile 7.4% / 51.2% / 0% collision numbers | differences explained by pairing/subset/metric, no contradiction | PASS | collision_metric_reconciliation.md | 7.4%=gate fac-vs-other on free (1 step); 51.2%=A1 all-pairs (=stuck rate, trivial); 0%=A1 fac-vs-CF on differing (2 steps). Consistent. |
