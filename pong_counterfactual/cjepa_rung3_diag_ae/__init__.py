"""Rung 3 reconstruction-autoencoder DIAGNOSTIC (read-only w.r.t. all frozen artifacts).

Narrow question: does a plain reconstruction-trained conv encoder preserve the Pong
player-paddle position and realized paddle displacement BETTER than the frozen
Discrete-JEPA tokenizer? Diagnostic only — trains no counterfactual transition model.

Everything here reuses the Rung 3 data contract (pong_counterfactual/cjepa_rung3):
same res84 frames, same fixed eval pool, same episode-grouped split, same oracle
branch cache. Nothing here writes to results/ or the cjepa_rung3 audit outputs.
"""
