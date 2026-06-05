"""CausalJEPA Rung 2 — abduction under the environment's OWN noise (ALE sticky actions).

Same abduction-vs-intervention test as Rung 1, but the stochasticity is no longer
self-injected wind: it is ALE's built-in sticky actions (repeat_action_probability=s).
The model still sees only observations; the seed-replay oracle pins the sticky outcome.
"""
