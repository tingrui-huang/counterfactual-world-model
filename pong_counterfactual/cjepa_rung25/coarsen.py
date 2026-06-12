"""Rung 2.5 coarsening operator: the stand-in for a lossy tokenizer.

The MODEL's whole world is quantized to bin width c px -- the state it conditions on
AND the delta-token vocabulary it predicts over -- while the ORACLE and the eval
targets stay full-fidelity (Invariant 1: never coarsen the measuring stick).

  coarsen(v, c) = floor(v / c) * c + c/2     (floor to the bin, decode to bin CENTER)

c = 1 is the IDENTITY: state_vec values pass through bit-unchanged, so the c=1 level
IS the unmodified Rung 2 setup and must reproduce its stored headline (the anchor,
acceptance check A). Per-level delta-bins are REFIT on the coarsened deltas: at c>=2
coarse states live on the c-lattice, so deltas are exact multiples of c and the
Discretizer with widths=(c,c,c,c) quantizes them losslessly. At c=1 we keep Rung 2's
DEFAULT_WIDTHS so the anchor is byte-identical.
"""
import numpy as np

from pong_counterfactual.cjepa_rung1.discretize import Discretizer, DEFAULT_WIDTHS
from pong_counterfactual.cjepa_rung1.noise import state_vec
from pong_counterfactual.cjepa_rung1.history import _onehot
from pong_counterfactual.cjepa_rung1.model import POS_NORM, VEL_NORM


def coarsen(vec, c):
    """Quantize a 4-vector to bin width c (px): floor to the bin, return the bin
    CENTER (physical units). c=1 is the identity (the Rung 2 anchor). None -> None."""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float64)
    if c == 1:
        return v
    return np.floor(v / c) * c + c / 2.0


def coarse_vecs(ep, c):
    """Per-episode list of coarsened 4-vectors (None where the state is invalid).
    Coarsening preserves None-ness, so valid_ks computed at full fidelity stays
    valid at every level (Invariant 2: same eval tuples everywhere)."""
    return [coarsen(state_vec(s), c) for s in ep.states]


def level_widths(c):
    """Delta-quantization widths for the level-c discretizer. c=1 keeps Rung 2's
    DEFAULT_WIDTHS (the anchor); c>=2 uses (c,c,c,c), exact on the c-lattice."""
    return DEFAULT_WIDTHS if c == 1 else (c, c, c, c)


def fit_level_discretizer(pairs, c):
    """Refit the delta-token vocabulary at level c from (coarse_state, coarse_next)
    pairs. Asserts every observed coarse training delta falls inside the refit bins
    (skill requirement: the refit must be lossless on what it was fit to)."""
    P = np.asarray([p for p, _ in pairs])
    P2 = np.asarray([q for _, q in pairs])
    disc = Discretizer(widths=level_widths(c)).fit(P, P2)
    cov = disc.coverage_report(P, P2)
    assert all(v == 1.0 for v in cov.values()), f"refit bins miss train deltas: {cov}"
    return disc


def hist_features_c(cvecs, intended, k, N, override_action=None):
    """history.hist_features, but over PRE-COARSENED vectors: the model's inputs live
    entirely in the coarse world. At c=1 this is bit-identical to Rung 2's features.
    override_action swaps ONLY the current intended action (the CF input)."""
    pos_k = cvecs[k]
    vel = pos_k - cvecs[k - 1]
    feat = list(pos_k / POS_NORM) + list(vel / VEL_NORM)
    feat += [cvecs[j][2] / POS_NORM for j in range(k - N + 1, k + 1)]
    acts = list(intended[k - N + 1:k + 1])
    if override_action is not None:
        acts[-1] = override_action
    for a in acts:
        feat += _onehot(a)
    return np.array(feat, dtype=np.float64)
