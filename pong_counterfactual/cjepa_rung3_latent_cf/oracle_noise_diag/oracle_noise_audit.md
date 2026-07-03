# Oracle-noise audit — the trusted environment-noise variable

## The stochastic mechanism (verified in the simulator code)

Pong here runs under **ALE sticky actions** (`repeat_action_probability=s`, s=0.5,
`env.py`). At each step the emulator, with probability s, **repeats the previously
executed action instead of the intended one** (the sticky draw is seeded by
`reset(seed)` and is action-independent — `env.py` docstring, verified by Rung-2's
`sticky_probe.py`). This action-independence is what makes seed replay pin the same
draw across an intervention.

## The trusted oracle-noise field: `sticky_fired` (two-probe `stuck`)

The most direct trusted simulator variable is the **two-probe sticky label**
(`oracle_rung2.Oracle.stuck_at`): replay to step k under the same seed, then step two
distinct actions p1,p2 (both ≠ a_prev); if the sticky fired BOTH execute a_prev →
identical next states → `stuck=1`, else they branch → `stuck=0`. It is computed by
**reset-replay** (the exact detector the whole Rung-2 line trusts) and is **never**
inferred from paddle movement. (A faster ALE `cloneSystemState` variant was tried and
REJECTED: it agreed with the trusted labels only 78% — not faithful.)

- **Values:** `u_oracle ∈ {0 = free, 1 = stuck}` (−1 = ill-posed probe; absent in the
  eval pool). Cardinality 2 — matches the parent model's `K=2`.
- **Relation to actions:** the *executed* action = `a_prev` if `stuck` else `a_int`.
  `sticky_fired` is the **exogenous** part; the executed action is `noise × (a_int,a_prev)`.
- We use `sticky_fired`, NOT `executed_action`, as the noise because it is the
  intervention-INVARIANT exogenous realization (see below); `executed_action` folds in
  the action and so is not reusable verbatim after an action swap.

## Coherent counterfactual meaning under action swap — YES

Because the ALE sticky draw is action-independent and seed-pinned, the SAME
`sticky_fired` realization holds when we swap the intended action a_int → a′:

- `stuck=1`: executed = `a_prev` regardless of a′ → the intervention is **nullified** →
  seed-replay CF next-state == factual next-state. (D0 confirms this at the latent level:
  on stuck eval tuples `z_cf_oracle_local == z_t1_local`.)
- `stuck=0`: executed = a′ → the paddle branches to the a′ move.

So reusing `u_oracle` under the swapped action is exactly what the seed-replay oracle
does — the noise is meaningfully reusable. This is the upper bound the parent
experiment's learned `u` only approximated (it matched `sticky_fired` ~59%).

## When noise values are observationally indistinguishable

`stuck=1` (executed a_prev) vs `stuck=0` (executed a_int) are indistinguishable exactly
when a_int and a_prev produce the same outcome:
- **a_int == a_prev** (no move-vs-stay contrast). In the eval pool a_prev is ALWAYS
  NOOP and a_int ∈ {UP,DOWN}, so a_int ≠ a_prev always → identifiable there.
- **boundary clamping**: if the paddle is against a wall and the intended move keeps it
  there, "move" and "stay" coincide → ambiguous. Reported as the boundary subset.

## Verdict
A trusted, simulator-derived noise variable with a coherent counterfactual meaning under
action swap EXISTS (`sticky_fired`). Proceed. Eval labels reuse the cached trusted
`stuck`; train labels are recomputed by the same reset-replay detector and cached,
aligned element-wise to the transition order (asserted in D0).
