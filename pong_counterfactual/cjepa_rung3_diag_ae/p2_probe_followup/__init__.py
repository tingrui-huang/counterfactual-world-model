"""P2 follow-up diagnostic (read-only).

Decides WHY the AE-latent displacement probe (P2) failed: was the temporal signal
discarded by the probe/pooling, or is it not decodably present in the frozen latent?
No retraining, no new encoder, no transition model — just alternative frozen-latent
probes on the SAME checkpoint / dataset / split / eval pool as the AE diagnostic.
"""
