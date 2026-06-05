"""Seed-replay oracle for Rung 2: ground-truth single-step counterfactuals under ALE's
OWN sticky-action noise.

The Rung-1 oracle pinned a RECORDED wind bit. Here there is no recorded bit: the noise
is ALE-internal. The oracle instead pins the sticky outcome by DETERMINISTIC SEED REPLAY
(validated by sticky_probe.py before any number is trusted):

    oracle_cf(seed, intended, k, a') = reset(seed); step intended[0..k-1]; step a'

Because the replay env runs with the SAME repeat_action_probability=s and the SAME seed
and the SAME prefix intended[0..k-1], its RNG reaches step k in the SAME state as the
factual run, so the sticky draw at k is IDENTICAL (Invariant 5). Hence:
  - if the action stuck at k, the executed action is the previous one regardless of a'
    => oracle_cf == factual next-state (the intervention was nullified);
  - if it did not stick, the executed action is a' => the counterfactual branches.
This holds the SAME sticky outcome fixed across the intervention -- exactly the noise
the model must infer by abduction.

The oracle env MUST be built with the same `s` as the episode it replays, or the sticky
draws will not match. `stuck_at` is an ORACLE-SIDE label (two-probe) used only to BIN
results for acceptance check E; it is NEVER fed to the model.
"""
import numpy as np

from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung1.noise import state_vec
from pong_counterfactual.cjepa_rung2.collect_rung2 import FRAMESKIP, NOOP, UP, DOWN

MOVES = [NOOP, UP, DOWN]


class Oracle:
    """Seed-replay oracle. Construct ONE per sticky level s (the replay env must match)."""

    def __init__(self, s):
        self.s = s
        # MUST match the collection frameskip AND sticky prob, or replay won't reproduce.
        self.env = PongEnv(frameskip=FRAMESKIP, repeat_action_probability=s)

    def _replay_step(self, seed, intended, k, a_k):
        """Replay intended[0:k] from seed, then step a_k once; return the 4-vector."""
        st = self.env.reset(seed=seed)
        for t in range(k):
            st, _, _ = self.env.step(int(intended[t]))
        st2, _, _ = self.env.step(int(a_k))
        return state_vec(st2)

    def cf_next_state(self, seed, intended, k, a_prime):
        """The counterfactual ground truth: replay to k, then take intended action a'.
        Seed replay automatically applies the SAME sticky outcome as the factual run."""
        return self._replay_step(seed, intended, k, a_prime)

    def factual_next_state(self, seed, intended, k):
        """No-op consistency / sanity: replay to k and take the FACTUAL action a_k.
        Under faithful seed replay this must equal the recorded factual next-state."""
        return self._replay_step(seed, intended, k, intended[k])

    def stuck_at(self, seed, intended, k):
        """ORACLE-SIDE label (NEVER a model input): did the action stick at step k?

        Two-probe detector: replay to k and step two distinct actions p1, p2, both
        != a_{k-1} and distinguishable in one step. If the sticky fired, BOTH execute
        a_{k-1} -> identical next-states -> stuck. Else they branch -> not stuck.
        Returns None if the probe is ill-posed (missing states)."""
        a_prev = intended[k - 1]
        cands = [a for a in MOVES if a != a_prev]
        p1, p2 = cands[0], cands[1]
        n1 = self._replay_step(seed, intended, k, p1)
        n2 = self._replay_step(seed, intended, k, p2)
        if n1 is None or n2 is None:
            return None
        return bool(np.allclose(n1, n2))

    def close(self):
        self.env.close()
