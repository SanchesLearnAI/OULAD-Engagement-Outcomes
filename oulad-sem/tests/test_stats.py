"""Tests for the shared statistics helpers."""
import numpy as np

from src.tma.stats import bh, holm


def test_bh_matches_a_hand_calculation():
    # Ten ascending p-values; the i-th smallest is scaled by n/i and then made
    # monotone from the largest p downwards.
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
         0.0459, 0.0500]
    adj = bh(p)
    expected = [0.0010, 0.0020, 0.0063, 0.0238, 0.0402, 0.0426, 0.0426,
                0.0430, 0.0500, 0.0500]
    assert np.allclose(adj, expected, atol=1e-4)
    assert np.all(adj >= np.asarray(p) - 1e-12)
    assert np.all(adj <= 1.0)


def test_bh_is_monotone_in_the_sorted_order():
    p = [0.5, 0.01, 0.3, 0.02]
    adj = bh(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_holm_is_at_least_as_conservative_as_bh():
    p = [0.001, 0.02, 0.04, 0.5]
    assert np.all(holm(p) >= bh(p) - 1e-12)


def test_holm_hand_calculation():
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.20]), [0.04, 0.09, 0.09, 0.20])
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_empty_and_single_inputs():
    assert len(bh([])) == 0
    assert len(holm([])) == 0
    assert np.allclose(bh([0.03]), [0.03])
    assert np.allclose(holm([0.03]), [0.03])
