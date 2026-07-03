"""Oracle-noise upper-bound diagnostic for the latent-space counterfactual experiment.

Question: given the TRUE environment noise (the trusted two-probe sticky label), can the
FROZEN paddle-local AE latent + small forward model produce accurate single-step
counterfactuals? Separates learned-noise-abduction failure from latent/forward failure.
Read-only w.r.t. all frozen artifacts; reuses the parent experiment's adapter + model.
"""
