"""Focused tests for the latent-CF experiment (adapter, model I/O alignment, CF tuples).
Run: python -m pytest pong_counterfactual/cjepa_rung3_latent_cf/test_latent_cf.py -q
"""
import numpy as np
import pytest
import torch

from pong_counterfactual.cjepa_rung3_latent_cf import latent_adapter as LA
from pong_counterfactual.cjepa_rung3_latent_cf import transition as TX

CFG = LA.load_config()
DATA = (LA._p(CFG["data_dir"]).exists()
        and (LA._p(CFG["data_dir"]) / "oracle_cache.npz").exists()
        and LA._p(CFG["ae_checkpoint"]).exists())
pytestmark = pytest.mark.skipif(not DATA, reason="AE ckpt or dataset not present")


# ---- adapter ----
def test_region_matches_p2_and_dim():
    from pong_counterfactual.cjepa_rung3_diag_ae.p2_probe_followup.run_followup import PADDLE_COLS
    assert list(CFG["paddle_cols"]) == [PADDLE_COLS.start, PADDLE_COLS.stop]
    ae, _ = LA.load_frozen_ae(CFG)
    b = LA.CM.load_eval_bundle(LA._p(CFG["data_dir"]))
    assert LA.encode_local(ae, b.cur_frame[:3]).shape == (3, 192)


def test_deterministic_encoding():
    ae, _ = LA.load_frozen_ae(CFG)
    b = LA.CM.load_eval_bundle(LA._p(CFG["data_dir"]))
    assert np.array_equal(LA.encode_local(ae, b.cur_frame[:16]),
                          LA.encode_local(ae, b.cur_frame[:16]))


def test_cf_tuple_alignment():
    """a_prime = opposite(a_int); on stuck tuples the oracle CF latent == factual latent."""
    ev = LA.build_eval_tuples(CFG)
    for ai, ap in zip(ev.a_int, ev.a_prime):
        assert {int(ai), int(ap)} == {LA.UP, LA.DOWN}
    same = np.all(np.isclose(ev.z_t1, ev.z_cf_oracle), axis=1)
    assert same[ev.stuck == 1].mean() > 0.95        # intervention nullified on stuck
    assert (~same[ev.stuck == 0]).mean() > 0.8       # branches on free


def test_train_eval_disjoint():
    tr = LA.build_train_transitions(CFG)
    ev = LA.build_eval_tuples(CFG)
    assert not (set(tr.ep_seed.tolist()) & set(ev.ep_seed.tolist()))


# ---- model I/O alignment ----
def test_forward_reacts_to_action_and_noise():
    torch.manual_seed(0)
    m = TX.LatentTransition(192, 8, K=2, hidden=64, depth=2)
    z = torch.randn(5, 192); a = torch.zeros(5, 8); a[:, 2] = 1  # UP
    a2 = torch.zeros(5, 8); a2[:, 3] = 1                          # DOWN
    u0 = torch.zeros(5, dtype=torch.long); u1 = torch.ones(5, dtype=torch.long)
    assert not torch.allclose(m(z, a, u0), m(z, a2, u0))          # action changes output
    assert not torch.allclose(m(z, a, u0), m(z, a, u1))           # noise changes output


def test_abduction_picks_lower_error_mode():
    """abduct must return the u minimizing the per-sample error to the observed next."""
    torch.manual_seed(1)
    m = TX.LatentTransition(192, 8, K=2, hidden=64, depth=2)
    z = torch.randn(7, 192); a = torch.zeros(7, 8); a[:, 2] = 1
    preds = m.predict_all_u(z, a)                                 # (7,2,192)
    # fabricate targets equal to a known u per row, abduction must recover it
    want = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    z_t1 = preds[torch.arange(7), want]
    u_hat, _ = m.abduct(z, a, z_t1)
    assert torch.equal(u_hat, want)


def test_forward_does_not_receive_next_latent():
    """Structural: forward signature takes (z_t, a, u) only — never z_t1."""
    import inspect
    params = list(inspect.signature(TX.LatentTransition.forward).parameters)
    assert params == ["self", "z_t", "a", "u_idx"]
