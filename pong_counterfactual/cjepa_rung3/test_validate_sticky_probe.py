"""Pure-helper tests for the Stage A1.1 sticky-probe validity audit.

Synthetic data only; never touches the real checkpoint/dataset.
Run:  python -m pytest pong_counterfactual/cjepa_rung3/test_validate_sticky_probe.py -q
"""
import numpy as np

from pong_counterfactual.cjepa_rung3 import validate_sticky_probe as V


def _separable(n=200, seed=0):
    """Two episode-grouped clusters whose label is linearly separable from X."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(20), n // 20)
    y = (groups % 2).astype(int)                       # label correlates with group
    X = (y[:, None] + 0.3 * rng.standard_normal((n, 4)))
    return X, y, groups


def test_grouped_auc_detects_signal():
    X, y, g = _separable()
    r = V.grouped_auc(X, y, g, repeats=10)
    assert r["auc_mean"] > 0.8 and r["n_splits"] > 0


def test_grouped_auc_chance_on_noise():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 4))
    groups = np.repeat(np.arange(20), 10)
    y = (np.arange(200) % 2)
    r = V.grouped_auc(X, y, groups, repeats=10)
    assert 0.3 < r["auc_mean"] < 0.7              # near chance for unrelated features


def test_permutation_test_flags_noise_as_nonsignificant():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((160, 4))
    groups = np.repeat(np.arange(16), 10)
    y = (np.arange(160) % 2)
    p = V.permutation_test_auc(X, y, groups, n_perm=60, repeats=4)
    assert p["p_value"] is not None and p["p_value"] > 0.05    # no real signal


def test_grouped_acc_reports_baseline():
    X, y, g = _separable()
    r = V.grouped_acc(X, y, g, repeats=10)
    assert r["acc_mean"] >= r["baseline"] - 0.05 and 0 <= r["baseline"] <= 1


def test_grouped_r2_runs():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(20), 10)
    X = rng.standard_normal((200, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.standard_normal(200)
    r = V.grouped_r2(X, y, groups, repeats=10)
    assert r["r2_mean"] > 0.9                      # cleanly linear -> high R2
