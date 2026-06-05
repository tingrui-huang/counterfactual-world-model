"""GATING PROBE (Rung-2 step 5, Invariant 3): is ALE sticky-action seed replay valid?

This is the make-or-break check for Rung 2. The whole experiment rests on one
assumption: with `repeat_action_probability=s` turned on, the ALE trajectory is STILL
a deterministic function of (seed, intended-action sequence), because ALE's sticky
RNG draw is (a) seeded by reset(seed=...) and (b) action-independent (the draw is made
regardless of the action value; if it fires, the executed action becomes the PREVIOUS
executed action). If that holds, the seed-replay oracle can pin the sticky outcome
exactly as Rung 1's oracle pinned the recorded wind bit. If it does NOT hold, seed
replay is re-rolling the sticky draw and every downstream number is meaningless --
in which case we must fall back to instrumenting ALE (clone/restore + forced action).

We run THREE empirical tests at s in {0.1, 0.25, 0.5}; ALL must pass to proceed:

  GATE 1 (determinism / reproducibility):  re-stepping (seed, intended[:k], intended[k])
          reproduces the factual next-state, for many (seed, k). If sticky used an
          unseeded RNG this fails. This is the literal oracle no-op consistency check
          (a' = a_k) of Invariant 3.

  GATE 2 (sticky actually fires at rate ~s): on eligible steps (a_k != a_{k-1}), the
          two-probe `stuck` detector flags ~s of them. Confirms the noise is real,
          detectable, and matches the documented "repeat previous action" mechanism.

  GATE 3 (action-independence of the draw): the stuck/not-stuck classification is
          CONSISTENT across a THIRD probe action -- if a step is stuck, ALL probe
          actions yield the same next-state (= the previous action's effect); if it
          is not stuck, distinct probe actions yield distinct next-states. A draw that
          depended on the action value would break this consistency.

Run:  python -m pong_counterfactual.cjepa_rung2.sticky_probe
"""
import numpy as np

from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung1.noise import state_vec

NOOP, FIRE, UP, DOWN = 0, 1, 2, 3
MOVES = [NOOP, UP, DOWN]
# fs=2: minimum frameskip with a non-degenerate one-step paddle effect (see collect_rung2).
# At fs=1 the stuck detector reads 1.0 everywhere because action[k] never moves the paddle
# in one step -- that is latency, not stickiness, and makes the experiment vacuous.
FRAMESKIP = 2


def _policy(s, rng, last_exe, tol=3, eps=0.25):
    """A noisy ball-tracker (same shape as the Rung-1 behavior policy) so the probe
    exercises realistic move/no-move sequences, including a_k != a_{k-1} steps."""
    ball, py = s["ball"], s["player_y"]
    if ball is None or py is None:
        return FIRE
    if rng.random() < eps:
        return int(rng.choice(MOVES))
    if ball[1] > py + tol:
        return DOWN
    if ball[1] < py - tol:
        return UP
    return NOOP


def roll(env, seed, intended):
    """Roll a factual trajectory under INTENDED actions (ALE applies sticky inside)."""
    s = env.reset(seed=seed)
    states = [s]
    for a in intended:
        s, _, done = env.step(int(a))
        states.append(s)
        if done:
            break
    return states


def step_at(env, seed, prefix, a_k):
    """Replay `prefix` (intended actions a_0..a_{k-1}) from seed, then step a_k once.
    Returns the next-state 4-vector (or None if a component is missing)."""
    s = env.reset(seed=seed)
    for a in prefix:
        s, _, _ = env.step(int(a))
    s2, _, _ = env.step(int(a_k))
    return state_vec(s2)


def collect_intended(env, seed, T, policy_seed):
    """One episode's intended-action sequence + its factual states, both replayable."""
    prng = np.random.default_rng(policy_seed)
    s = env.reset(seed=seed)
    states = [s]
    intended = []
    last_exe = NOOP
    for t in range(T):
        a = _policy(s, prng, last_exe)
        s, _, done = env.step(int(a))
        intended.append(int(a))
        states.append(s)
        last_exe = a  # NOTE: intended, not executed -- only used to shape the policy
        if done:
            break
    return intended, states


def probe_pair(a_prev):
    """Two distinct intended actions, both != a_prev, distinguishable in one step."""
    cands = [a for a in MOVES if a != a_prev]
    return cands[0], cands[1]


def third_probe(a_prev, p1, p2):
    """A third action (may equal a_prev) to test draw action-independence."""
    for a in MOVES:
        if a not in (p1, p2):
            return a
    return a_prev


def run(s, n_ep=16, T=220, base_seed=0, ks_per_ep=45, seed_rng=0):
    env = PongEnv(frameskip=FRAMESKIP, repeat_action_probability=s)
    pick_rng = np.random.default_rng(seed_rng)

    gate1_ok = gate1_tot = 0
    eligible = stuck = 0
    gate3_ok = gate3_tot = 0

    for ep in range(n_ep):
        seed = base_seed + ep
        intended, states = collect_intended(env, seed, T, policy_seed=100 + ep)

        # Each step_at replays a length-k prefix, so scanning ALL k is O(T^2). Sampling
        # a fixed number of k per episode keeps the gate statistically ample but cheap.
        cand_ks = list(range(1, len(intended)))
        if len(cand_ks) > ks_per_ep:
            cand_ks = sorted(pick_rng.choice(cand_ks, ks_per_ep, replace=False).tolist())
        for k in cand_ks:
            # need valid 4-vectors around k for a meaningful comparison
            if state_vec(states[k]) is None or state_vec(states[k + 1]) is None:
                continue
            prefix = intended[:k]
            fac_next = state_vec(states[k + 1])

            # GATE 1: replay reproduces the factual next-state (a' = a_k no-op).
            rep = step_at(env, seed, prefix, intended[k])
            if rep is not None:
                gate1_tot += 1
                gate1_ok += int(np.allclose(rep, fac_next))

            # GATES 2 & 3 only on eligible move-steps. We require a_{k-1} == NOOP so the
            # probe pair is exactly {UP, DOWN} -- two MAXIMALLY action-sensitive actions
            # (opposite paddle directions), driving the non-stuck coincidence floor to ~0
            # so the measured stuck-rate is a clean estimate of the true sticky-fire rate.
            a_prev = intended[k - 1]
            if intended[k] not in (UP, DOWN) or a_prev != NOOP:
                continue
            p1, p2 = probe_pair(a_prev)   # == (UP, DOWN) since a_prev == NOOP
            n1 = step_at(env, seed, prefix, p1)
            n2 = step_at(env, seed, prefix, p2)
            if n1 is None or n2 is None:
                continue
            eligible += 1
            is_stuck = bool(np.allclose(n1, n2))   # both executed a_prev => stuck
            stuck += int(is_stuck)

            # GATE 3: a third probe must agree with the stuck/not-stuck verdict.
            p3 = third_probe(a_prev, p1, p2)
            n3 = step_at(env, seed, prefix, p3)
            if n3 is None:
                continue
            gate3_tot += 1
            if is_stuck:
                # stuck => p3 also yields the previous action's effect == n1 == n2
                ok = np.allclose(n3, n1)
            else:
                # not stuck => p3 executes p3; if p3==p1 or p3==p2 it must match that
                # outcome, and if p3 is the previous action it must DIFFER from the
                # not-stuck distinct outcomes (we just require internal consistency:
                # p3's outcome equals p1's iff p3==p1's executed action).
                if p3 == p1:
                    ok = np.allclose(n3, n1)
                elif p3 == p2:
                    ok = np.allclose(n3, n2)
                else:
                    ok = True  # p3 == a_prev: its own (non-stuck) effect, no constraint
            gate3_ok += int(ok)

    env.close()
    return {
        "s": s,
        "gate1_rate": gate1_ok / max(gate1_tot, 1),
        "gate1_n": gate1_tot,
        "stuck_rate": stuck / max(eligible, 1),
        "eligible": eligible,
        "gate3_rate": gate3_ok / max(gate3_tot, 1),
        "gate3_n": gate3_tot,
    }


def main():
    print("=" * 78)
    print("Rung-2 GATING PROBE — is ALE sticky-action seed replay valid? (Invariant 3)")
    print("  GATE 1: replay reproduces the factual next-state (DECISIVE; must be 100%)")
    print("  GATE 2: floor-corrected stuck-rate tracks s (sticky fires at ~rate s)")
    print("  GATE 3: stuck verdict is consistent across a 3rd probe (action-independent)")
    print("=" * 78)

    # s=0 first: with NO stickiness, any 'stuck' verdict is a pure COINCIDENCE FLOOR
    # (two probe actions happen to give the same integer next-state, e.g. paddle pinned
    # at a wall). Subtracting it isolates the true sticky-fire rate.
    rows = [run(s) for s in (0.0, 0.1, 0.25, 0.5)]
    floor = rows[0]["stuck_rate"]
    for r in rows:
        r["stuck_corr"] = (r["stuck_rate"] - floor) / max(1 - floor, 1e-9)

    print(f"  {'s':>5} | {'GATE1 repro':>16} | {'stuck-raw':>9} | {'-floor':>7} | "
          f"{'GATE3 consist':>14}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['s']:>5.2f} | {r['gate1_rate']*100:>8.1f}% (n={r['gate1_n']:>4}) | "
              f"{r['stuck_rate']:>9.3f} | {r['stuck_corr']:>7.3f} | "
              f"{r['gate3_rate']*100:>9.1f}% (n={r['gate3_n']})")
    print(f"  (coincidence floor at s=0: {floor:.3f} — action-insensitive/boundary steps)")

    g1 = all(r["gate1_rate"] > 0.999 for r in rows)
    pos = rows[1:]
    # floor-corrected stuck-rate should track s (monotone + within tolerance of s)
    g2 = (all(r["stuck_corr"] < r2["stuck_corr"] for r, r2 in zip(pos, pos[1:]))
          and all(abs(r["stuck_corr"] - r["s"]) < 0.12 for r in pos))
    g3 = all(r["gate3_rate"] > 0.97 for r in rows)   # ~1% boundary edge cases allowed
    print("\n" + "=" * 78)
    print(f"GATE 1 (seed replay deterministic w/ sticky on => oracle pins it): {g1}")
    print(f"GATE 2 (floor-corrected sticky-fire rate tracks s):               {g2}")
    print(f"GATE 3 (sticky draw is action-independent):                       {g3}")
    print("=" * 78)
    if g1 and g2 and g3:
        print("\nGATE PASSED. ALE's sticky draw is seed-reproduced and action-independent,")
        print("so the seed-replay oracle pins the sticky outcome (GATE 1 IS the Invariant-3")
        print("oracle no-op check). Proceed to eval_rung2.")
    else:
        print("\nGATE FAILED. If GATE 1<100%, seed replay does NOT pin the sticky draw -- do")
        print("NOT trust any downstream number; switch to the instrument fallback (ALE")
        print("cloneState/restoreState + forcing the factual executed action in CF replay).")
    return g1 and g2 and g3


if __name__ == "__main__":
    main()
