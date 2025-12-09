import numpy as np
import pytest

from ctfeval.ctf import get_ctf_source_cov, spurious_coherence, ctf_ratio_cov


@pytest.mark.parametrize(
    "source_cov,expected",
    [
        (np.eye(2), np.array([[0.36], [0.64]])),
        (np.diag([2, 1]), np.array([[0.53], [0.47]])),
    ],
)
def test_get_ctf_source_cov(source_cov, expected):
    ctf_orig = np.array([[3], [4]])
    assert np.allclose(get_ctf_source_cov(ctf_orig, source_cov), expected, atol=0.01)


@pytest.mark.parametrize(
    "ctf1,ctf2,source_cov,expected_sc",
    [
        # no overlap -> no SC
        ([1, 0], [0, 1], None, 0),
        # SC is symmetric if all source amplitudes are equal
        ([1, 1], [0, 1], None, np.sqrt(0.5)),
        ([1, 0], [1, 1], None, np.sqrt(0.5)),
        # Source amplitudes play a role
        ([1, 1], [0, 1], [[1, 0], [0, 4]], np.sqrt(0.8)),
        ([1, 0], [1, 1], [[1, 0], [0, 4]], np.sqrt(0.2)),
        # Source correlation don't play a role
        ([1, 1], [0, 1], [[1, -1], [1, 4]], np.sqrt(0.8)),
        ([1, 0], [1, 1], [[1, -1], [1, 4]], np.sqrt(0.2)),
    ],
)
def test_spurious_coherence(ctf1, ctf2, source_cov, expected_sc):
    if source_cov is not None:
        source_cov = np.array(source_cov)
    sc = spurious_coherence(np.array(ctf1), np.array(ctf2), source_cov)
    assert np.isclose(sc, expected_sc)


@pytest.mark.parametrize(
    "w,source_cov,data_cov,expected",
    [
        (np.array([1, 0]), None, "auto", 1.0),
        (np.array([1, 1]), None, "auto", 0.5),
        (np.array([0, 1]), None, "auto", 0.0),
        (np.array([1, 1]), np.diag([1, 4]), "auto", 0.2),
        (np.array([1, 1]), np.diag([4, 1]), "auto", 0.8),
        (np.array([1, 1]), None, np.diag([1, 4]), 0.2),
        (np.array([1, 1]), None, np.diag([4, 1]), 0.2),
    ],
)
def test_ctf_ratio_cov(w, source_cov, data_cov, expected):
    L = np.eye(2)

    # Copied from src/evaluate.py:evaluate_theory
    data_cov_default = L @ source_cov @ L.T if source_cov is not None else L @ L.T

    mask = np.array([True, False])
    rat = ctf_ratio_cov(w, L, mask, data_cov_default, source_cov, data_cov)

    assert np.allclose(rat, expected)
