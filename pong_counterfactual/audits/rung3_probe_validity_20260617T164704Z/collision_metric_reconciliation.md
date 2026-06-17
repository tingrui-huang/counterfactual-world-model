# Collision metric reconciliation

Three different collision numbers appear across the audits. They are **not** in conflict — they differ in dataset pairing, subset, and the physical magnitude of the compared difference.

| number | pairing | subset | meaning |
| --- | --- | --- | --- |
| **7.4%** (Stage-2 gate) | factual-next vs OTHER (=step a_prev, the would-have-stuck branch; ~1 paddle step) | FREE tuples (n=122) | the gate's collision; recomputed here = 0.074 |
| **51.2%** (A1 R2 all-pairs) | factual vs CF (opposite intended; ~2 paddle steps) | ALL 250 | dominated by the 128 stuck pairs which are physically IDENTICAL frames (collide trivially) -> equals the stuck rate, NOT a separability failure |
| **0.0%** (A1 R2 physically-different) | factual vs CF | physically-different pairs | recomputed = 0.000; the eye keeps the larger 2-step difference fully distinct |

**Reconciliation:** 7.4% (1-step, fac-vs-other) > 0.0% (2-step, fac-vs-CF) is expected — a 2-step opposite-action difference is easier to keep distinct than a 1-step would-have-stuck difference. 51.2% is an all-pairs artifact (the stuck rate), not a representation metric. All three are mutually consistent.
