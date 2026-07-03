"""Rung 3 latent-space counterfactual experiment.

Transfers the Rung 2 abduction/CF-vs-IV pipeline from ground-truth object state to the
FROZEN paddle-local AE latent, WITHOUT decoding the latent back into object coordinates.
Discrete-noise SCM (mixture) forward model + argmin abduction; seed-replay oracle reused
via the AE diagnostic's cached CF frames. Read-only w.r.t. all frozen artifacts.
See design_audit.md for the noise-adaptation justification.
"""
