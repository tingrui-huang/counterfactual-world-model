"""Pure-helper tests for the Stage A1 representation abductability audit.

These exercise the audit's deterministic helpers (one-hot, physical distances, episode-
grouped split, executed-action logic, bootstrap CI, baselines) on tiny synthetic data.
They do NOT touch the real checkpoint or any dataset.

Run:  python -m pytest pong_counterfactual/cjepa_rung3/test_diagnose_representation.py -q
"""
import numpy as np

from pong_counterfactual.cjepa_rung3 import diagnose_representation as D


def test_onehot_codes_shape_and_blocks():
    codes = np.array([[0, 1], [2, 0]])
    X = D.onehot_codes(codes, M=2, K=3)            # (2, 6) slot-blocked
    assert X.shape == (2, 6)
    # row0: slot0->code0 (col0), slot1->code1 (col 3+1=4)
    assert X[0, 0] == 1.0 and X[0, 4] == 1.0 and X[0].sum() == 2
    # row1: slot0->code2 (col2), slot1->code0 (col3)
    assert X[1, 2] == 1.0 and X[1, 3] == 1.0


def test_physical_distances_identity_and_diff():
    o_fac = np.array([[10., 20., 30., 40.], [1., 2., 3., 4.]])
    o_cf = np.array([[10., 20., 30., 40.], [1., 2., 9., 4.]])   # 2nd differs in player_y
    other = o_fac.copy()
    p = D.physical_distances(o_fac, o_cf, other)
    assert p["identical_fac_cf"][0] and not p["identical_fac_cf"][1]
    assert abs(p["dplayer_y_fac_cf"][1] - 6.0) < 1e-9
    assert p["identical_fac_oth"].all()
    assert p["valid_fac_cf"].all()


def test_physical_distances_nan_invalidates():
    o_fac = np.array([[np.nan, 1., 2., 3.]])
    o_cf = np.array([[0., 1., 2., 3.]])
    p = D.physical_distances(o_fac, o_cf, o_cf.copy())
    assert not p["valid_fac_cf"][0]
    assert not p["identical_fac_cf"][0]            # invalid -> not counted identical


def test_group_split_no_episode_leakage():
    groups = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    tr, te = D.group_train_test_split(groups, test_frac=0.5, seed=0)
    assert (tr & te).sum() == 0 and (tr | te).all()
    # every episode is wholly in one side
    for g in np.unique(groups):
        gm = groups == g
        assert tr[gm].all() or te[gm].all()


def test_executed_action_sticky_logic():
    a_int = np.array([2, 3, 2])
    a_prev = np.array([0, 2, 3])
    stuck = np.array([1, 0, -1])                   # fired / not / ill-posed
    exe = D.executed_action(a_int, a_prev, stuck)
    assert exe.tolist() == [0, 3, 2]               # stuck->prev, else intended


def test_bootstrap_ci_brackets_mean():
    vals = np.concatenate([np.ones(50), np.zeros(50)])   # mean 0.5
    lo, hi = D.bootstrap_ci(vals, n_boot=500, seed=0)
    assert lo < 0.5 < hi and 0.0 <= lo <= hi <= 1.0


def test_majority_baseline():
    ytr = np.array([1, 1, 1, 0])
    yte = np.array([1, 1, 0, 0])
    assert D._majority_acc(ytr, yte) == 0.5        # predicts 1 -> 2/4 correct


def test_oh_one_hot():
    X = D._oh(np.array([0, 2, 1]), 4)
    assert X.shape == (3, 4)
    assert X[0, 0] == 1 and X[1, 2] == 1 and X[2, 1] == 1


def test_near_chance_control():
    assert D._near_chance({"shuffled_control": 0.52, "baseline": 0.5})
    assert not D._near_chance({"shuffled_control": 0.9, "baseline": 0.5})
    assert D._near_chance({"shuffled_control": "insufficient samples", "baseline": 0.5})
