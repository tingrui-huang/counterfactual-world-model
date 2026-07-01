"""Focused tests for the reconstruction-AE diagnostic: data contract, model forward,
split integrity, probe behavior. Pure/fast — no training, no frozen-artifact writes.

Run:  python -m pytest pong_counterfactual/cjepa_rung3_diag_ae/test_diag_ae.py -q
"""
import numpy as np
import pytest

from pong_counterfactual.cjepa_rung3_diag_ae import common as CM
from pong_counterfactual.cjepa_rung3_diag_ae import probes as PB
from pong_counterfactual.cjepa_rung3_diag_ae.ae_model import (
    ConvAutoencoder, AEConfig, latent_dim,
)

DATA_PRESENT = CM.DEFAULT_DATA_DIR.exists() and (
    CM.DEFAULT_DATA_DIR / "oracle_cache.npz").exists()


# ------------------------------------------------------------------- model forward
def test_ae_forward_shapes():
    m = ConvAutoencoder(AEConfig())
    import torch
    x = torch.randint(0, 256, (7, 84, 84), dtype=torch.uint8)
    loss, recon, z = m.recon_loss(x)
    assert recon.shape == (7, 1, 84, 84)
    assert z.shape == (7, 16, 6, 6)
    assert float(loss) >= 0.0


def test_ae_encode_numpy_dim():
    cfg = AEConfig()
    m = ConvAutoencoder(cfg)
    x = np.random.randint(0, 256, (5, 84, 84)).astype(np.uint8)
    z = m.encode_numpy(x)
    assert z.shape == (5, latent_dim(cfg))
    assert z.dtype == np.float32


def test_ae_loss_selectable():
    import torch
    x = torch.randint(0, 256, (4, 84, 84), dtype=torch.uint8)
    for kind in ("mse", "l1"):
        m = ConvAutoencoder(AEConfig(loss=kind))
        loss, _, _ = m.recon_loss(x)
        assert np.isfinite(float(loss))


# ------------------------------------------------------------------- split integrity
def test_group_split_no_overlap_and_covers():
    groups = np.repeat(np.arange(14), 20)  # 14 episodes, 20 tuples each
    tr, te = CM.group_train_test_split(groups, test_frac=0.3, seed=0)
    assert (tr | te).all() and not (tr & te).any()          # partition
    assert not (set(groups[tr]) & set(groups[te]))          # no group leaks across
    assert set(groups[tr]) | set(groups[te]) == set(groups)  # every group placed


def test_group_split_deterministic():
    g = np.repeat(np.arange(10), 5)
    a = CM.group_train_test_split(g, 0.3, seed=0)
    b = CM.group_train_test_split(g, 0.3, seed=0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ------------------------------------------------------------------- probes
def _synthetic(n=400, d=8, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y = X @ w + noise * rng.normal(size=n)
    groups = np.arange(n) % 12
    tr, te = CM.group_train_test_split(groups, 0.3, seed=0)
    return X, y, tr, te


def test_reg_probe_recovers_signal_and_shuffle_control():
    X, y, tr, te = _synthetic()
    r = PB.reg_probe(X, y, tr, te, model_name="ridge", seed=0)
    # a real linear signal must beat both the mean baseline and the shuffled control
    assert r["test_mae"] < 0.5 * r["baseline_mae"]
    assert r["test_mae"] < 0.5 * r["shuffled_mae"]
    assert r["test_r2"] > 0.8


def test_reg_probe_pure_noise_no_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = rng.normal(size=300)                     # independent of X
    groups = np.arange(300) % 10
    tr, te = CM.group_train_test_split(groups, 0.3, seed=0)
    r = PB.reg_probe(X, y, tr, te, seed=0)
    # no signal: test MAE should be close to the shuffled control (within 25%)
    assert r["test_mae"] > 0.7 * r["shuffled_mae"]


def test_clf_probe_recovers_and_baseline():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int)                # separable
    groups = np.arange(300) % 10
    tr, te = CM.group_train_test_split(groups, 0.3, seed=0)
    r = PB.clf_probe(X, y, tr, te, seed=0)
    assert r["test_acc"] > 0.9
    assert r["shuffled_acc"] <= r["baseline_acc"] + 0.15


def test_error_by_bin_partitions_all():
    pred = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    rows = PB.error_by_bin(pred, y, n_bins=5)
    assert sum(r["n"] for r in rows) == 50
    assert all(r["mae"] < 1e-9 for r in rows if r["n"])   # perfect prediction


# ------------------------------------------------------------------- data contract
@pytest.mark.skipif(not DATA_PRESENT, reason="res84 dataset/oracle not present")
def test_eval_bundle_delta_and_branches():
    b = CM.load_eval_bundle()
    assert len(b.pool) == 250
    assert np.allclose(b.delta_y, b.py_k1 - b.py_k)
    # branch counts: identical (stuck) + different (free) == valid pairs
    valid = b.phys["valid_fac_cf"]
    ident = b.phys["identical_fac_cf"]
    assert int((valid & ident).sum()) + int((valid & ~ident).sum()) == int(valid.sum())
    # no NaN in the probe targets
    assert not np.isnan(b.py_k).any() and not np.isnan(b.py_k1).any()


@pytest.mark.skipif(not DATA_PRESENT, reason="res84 dataset/oracle not present")
def test_train_eval_episode_disjoint():
    _, seeds, _ = CM.load_train_frames()
    b = CM.load_eval_bundle()
    assert not (set(seeds.tolist()) & set(b.ep_seed.tolist()))
