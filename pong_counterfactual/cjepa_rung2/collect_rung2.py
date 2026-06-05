"""Collect offline Pong transitions under ALE's OWN sticky-action noise (Rung 2).

The ONLY change from Rung 1's collection is the noise SOURCE:
  Rung 1: self-injected NOOP-override wind, recorded as a bool array.
  Rung 2: ALE sticky actions (repeat_action_probability=s) -- internal, seed-determined,
          NEVER recorded and NEVER shown to the model. Injected wind is OFF.

So we log ONLY what the model may see plus what the oracle needs to REPLAY:
  - seed                      (oracle only)
  - intended[]                (oracle replays these; model sees them as its action input)
  - states[]                  (model-facing observations)
There is no `executed[]` and no `wind[]`: the sticky outcome lives inside ALE and is
recovered purely by deterministic seed replay (validated by sticky_probe.py).

Because the sticky override target is the PREVIOUS action (not a fixed NOOP), the
next-state given (state, intended_action) is no longer a function of the model's inputs
unless it can see the recent action history -- so history(N) is LOAD-BEARING here
(it was merely helpful in Rung 1.1). We reuse history.py.

Behavior policy: the same noisy ball-tracker + tap discipline as Rung 1, but its
tap-release keys off the last INTENDED action (we never observe the executed one).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung1.noise import state_vec
from pong_counterfactual.cjepa_rung1.history import valid_ks

NOOP, FIRE, UP, DOWN = 0, 1, 2, 3
ACTIONS = [NOOP, UP, DOWN]          # the intended-action vocabulary the model sees
# Rung 2 DEVIATES from Rung 1's frameskip=1. Empirically (sticky_probe / _tmp_paddle),
# at fs=1 the Pong paddle has a 1-FRAME ACTUATION LATENCY: action[k] does NOT move
# player_y until k+2 (player_y[k+1] = player_y[k] + velocity[k], and velocity[k] is set
# by action[k-1]). So at fs=1 a SINGLE-STEP paddle counterfactual is DEGENERATE -- the
# intervention cannot change the observed next-state (action[k] effect on state[k+1] = 0%).
# fs=2 is the MINIMUM frameskip giving a one-step action effect (100%), with the fewest
# hidden sub-step frames -> cleanest s=0 Markov control. The residual 2-frame paddle
# momentum is exactly what history(N) (load-bearing here) absorbs; check C validates it.
FRAMESKIP = 2


def behavior_policy(s, rng, last_int=NOOP, tol=3, eps=0.2):
    """Noisy ball-tracker with a TAP discipline (no two intended moves in a row), keyed
    on the last INTENDED action since the executed one is hidden under sticky. Returns an
    intended action in {NOOP, UP, DOWN}, or FIRE to serve when the ball is absent."""
    ball, py = s["ball"], s["player_y"]
    if ball is None or py is None:
        return FIRE
    if last_int in (UP, DOWN):
        return NOOP                  # tap-release: keeps the paddle slow / low-momentum
    if rng.random() < eps:
        return int(rng.choice(ACTIONS))
    target = ball[1]
    if target > py + tol:
        return DOWN
    if target < py - tol:
        return UP
    return NOOP


@dataclass
class Episode:
    seed: int
    intended: List[int]             # full per-step intended actions (oracle replays these)
    states: List                    # state DICTS, length len(intended)+1
    # NOTE: no `executed`, no `wind` -- the sticky outcome is internal to ALE.
    valid_idx: List[int] = field(default_factory=list)  # move-steps (legacy, N-agnostic)


def collect(s, n_episodes, T=250, base_seed=0, policy_seed=0):
    """Roll n_episodes with ALE sticky actions at probability s. Returns list[Episode]."""
    env = PongEnv(frameskip=FRAMESKIP, repeat_action_probability=s)
    prng = np.random.default_rng(policy_seed)
    episodes = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        st = env.reset(seed=seed)
        states = [st]
        intended = []
        last_int = NOOP
        for t in range(T):
            a_int = behavior_policy(st, prng, last_int)
            st2, _, done = env.step(int(a_int))   # ALE applies sticky internally
            intended.append(int(a_int))
            states.append(st2)
            st = st2
            last_int = a_int
            if done:
                break
        # A simple move-step validity list (N-agnostic); training/eval use history.valid_ks
        # for the proper N-window check. Kept for quick stats only.
        valid = []
        for t in range(1, len(intended)):
            if intended[t] not in (UP, DOWN):
                continue
            if (state_vec(states[t - 1]) is None or state_vec(states[t]) is None
                    or state_vec(states[t + 1]) is None):
                continue
            valid.append(t)
        episodes.append(Episode(seed, intended, states, valid))
    env.close()
    return episodes


# re-export for convenience
__all__ = ["Episode", "collect", "valid_ks", "ACTIONS", "NOOP", "UP", "DOWN", "FRAMESKIP"]
