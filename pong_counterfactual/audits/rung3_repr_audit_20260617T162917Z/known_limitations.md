# Known limitations

- In this eval pool a_prev is ALWAYS NOOP (True): the behavior policy precedes every move with a NOOP, so a sticky fire repeats NOOP and the realized-noise signal is exactly 'paddle STAYED PUT vs moved ~1 paddle step (~4.6 screen px)'. That 1-step difference is near the M4_K64_res84 eye's KNOWN coarse paddle localization limit (its Stage-2 probe failed at player_y MAE 4.05 resized px), which is the most likely reason z_{t+1} fails to add a clean sticky signal — the move-vs-stay difference is at/below token resolution.
- Eval coverage is the fixed 250-tuple pool over 14 eval episodes; episode-grouped splits leave few test episodes, so probe CIs are wide — treat R3 effect sizes as indicative, not definitive.
- Sticky is identifiable only when a_prev≠a_int; the identifiable subset is smaller than the full pool, further limiting probe power.
- Physical 'pixel difference' uses the preprocessed res128 frames; sub-pixel paddle moves can vanish under the bilinear resize (the known Rung-3 ① issue).
- z_{t+1} is the FACTUAL next latent only; this audit does not encode the counterfactual branch into a transition — it asks whether the factual transition is informative, a necessary (not sufficient) condition for abduction.
- Probes are linear/logistic by design; a negative result bounds LINEAR decodability, not all decodability.
