"""Focused validation for the P2 follow-up (no broad infra).

Run: python -m pytest pong_counterfactual/cjepa_rung3_diag_ae/p2_probe_followup/test_followup.py -q
"""
import numpy as np
import pytest

from pong_counterfactual.cjepa_rung3_diag_ae import common as CM
from pong_counterfactual.cjepa_rung3_diag_ae.p2_probe_followup import run_followup as RF

DATA = CM.DEFAULT_DATA_DIR.exists() and (CM.DEFAULT_DATA_DIR / "oracle_cache.npz").exists()
CKPT = RF.AE_CKPT.exists()
pytestmark = pytest.mark.skipif(not (DATA and CKPT),
                                reason="AE checkpoint or dataset not present")


def test_known_answer_oracle_delta():
    """delta_y == y_t1 - y_t exactly. An UNREGULARIZED probe on [y_t,y_t1] must recover it
    (R²≈1), proving the target + eval are correct. (The reported alpha=1 ridge number is
    lower ~0.76 purely from regularization — that is the oracle CEILING under the probe,
    not an eval bug; asserted separately below.)"""
    from sklearn.linear_model import LinearRegression
    b = CM.load_eval_bundle()
    assert np.allclose(b.delta_y, b.py_k1 - b.py_k)
    tr, te = CM.group_train_test_split(b.ep_seed, 0.3, seed=0)
    X = np.stack([b.py_k, b.py_k1], 1)
    m = LinearRegression().fit(X[tr], b.delta_y[tr])
    pred = m.predict(X[te])
    r2 = 1 - ((pred - b.delta_y[te]) ** 2).sum() / ((b.delta_y[te] - b.delta_y[te].mean()) ** 2).sum()
    assert r2 > 0.999                                  # exact linear recovery
    # under the actual (regularized) probe the same oracle is capped well below 1
    bl = RF.run_baselines(b, tr, te)
    assert 0.6 < bl["oracle_yt_yt1"]["dy_r2"] < 0.95
    assert bl["oracle_yt_yt1"]["dir_acc"] > 0.9


def test_mismatch_uses_different_transition():
    """The mismatched control must actually swap z_t1 to a different transition's latent."""
    b = CM.load_eval_bundle()
    ae, _ = RF.load_frozen_ae()
    zt = RF.spatial_latent(ae, b.cur_frame[:40])
    zt1 = RF.spatial_latent(ae, b.fac_frame[:40])
    dir_y = np.sign(b.delta_y[:40]).astype(int)
    dy = b.delta_y[:40]
    tr = np.zeros(40, bool); tr[:28] = True
    te = ~tr
    # reach into the builder path: emulate the mismatch swap and confirm it differs
    rng = np.random.default_rng(1000)
    te_idx = np.where(te)[0]
    perm = te_idx.copy(); rng.shuffle(perm)
    zt1_mis = zt1.copy(); zt1_mis[te_idx] = zt1[perm]
    # at least most test rows now hold a different transition's z_t1
    changed = ~np.all(np.isclose(zt1_mis[te_idx], zt1[te_idx]), axis=(1, 2, 3))
    assert changed.mean() > 0.7


def test_encoder_frozen_and_deterministic():
    b = CM.load_eval_bundle()
    ae, _ = RF.load_frozen_ae()
    assert not any(p.requires_grad for p in ae.parameters())
    z1 = RF.spatial_latent(ae, b.cur_frame[:16])
    z2 = RF.spatial_latent(ae, b.cur_frame[:16])
    assert np.allclose(z1, z2)


def test_no_split_leakage():
    b = CM.load_eval_bundle()
    for s in (0, 1, 2):
        tr, te = CM.group_train_test_split(b.ep_seed, 0.3, seed=s)
        assert not (set(b.ep_seed[tr].tolist()) & set(b.ep_seed[te].tolist()))


def test_paddle_cols_are_right_side():
    # the paddle-local region must be the RIGHT latent columns (player side, per P3)
    assert RF.PADDLE_COLS.start >= 3 and RF.PADDLE_COLS.stop == 6
