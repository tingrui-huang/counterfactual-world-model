"""Focused tests for the oracle-noise diagnostic.
Run: python -m pytest pong_counterfactual/cjepa_rung3_latent_cf/oracle_noise_diag/test_oracle_noise.py -q
"""
import numpy as np
import pytest

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf.oracle_noise_diag import oracle_noise as ON

CFG = ON.load_config()
DATA = (LA._p(CFG["data_dir"]).exists()
        and (LA._p(CFG["data_dir"]) / "oracle_cache.npz").exists()
        and LA._p(CFG["ae_checkpoint"]).exists())
CACHE = ON.CACHE.exists()
pytestmark = pytest.mark.skipif(not DATA, reason="AE ckpt/dataset absent")


def test_eval_noise_is_binary_from_cache():
    ev = LA.build_eval_tuples(CFG)
    assert set(np.unique(ev.stuck)) <= {0, 1}          # trusted cached two-probe label
    assert (ev.stuck >= 0).all()


def test_cf_meaning_under_swap():
    """On stuck tuples the seed-replay CF latent equals the factual next latent."""
    ev = LA.build_eval_tuples(CFG)
    same = np.all(np.isclose(ev.z_t1, ev.z_cf_oracle), axis=1)
    assert same[ev.stuck == 1].mean() > 0.95
    assert (~same[ev.stuck == 0]).mean() > 0.8


def test_wrong_and_shuffled_noise_are_incorrect():
    ev = LA.build_eval_tuples(CFG)
    wrong = 1 - ev.stuck
    assert (wrong != ev.stuck).all()                   # wrong-noise genuinely flips
    rng = np.random.default_rng(0)
    shuf = ev.stuck[rng.permutation(len(ev.stuck))]
    assert (shuf != ev.stuck).mean() > 0.3             # shuffled genuinely differs


@pytest.mark.skipif(not CACHE, reason="train stuck cache not built")
def test_train_noise_aligned_and_rate():
    train = LA.build_train_transitions(CFG)
    stuck, key = ON.build_train_stuck(CFG)
    ON.assert_alignment(train, key)                     # raises if misaligned
    assert 0.35 < (stuck[stuck >= 0] == 1).mean() < 0.65   # ~ sticky s=0.5


def test_noise_field_is_simulator_not_movement():
    """Guard: the builder uses the two-probe Oracle, not a paddle-movement heuristic."""
    import inspect
    src = inspect.getsource(ON.build_train_stuck)
    assert "stuck_at" in src and "Oracle" in src
    assert "py_k1" not in src and "delta" not in src.lower()
