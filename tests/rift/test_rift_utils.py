import numpy as np
import pytest

from ctfeval.rift.utils import (
    preproc_onestim,
    preproc_twostim,
    preproc_delta,
    matrix_from_triu,
)


def test_preproc_onestim_no_threshold():
    coh_real_orig = np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (3, 1))
    coh_theory_orig = coh_real_orig

    coh_theory, coh_real, consider_all = preproc_onestim(coh_theory_orig, coh_real_orig)
    assert coh_theory.shape[0] == coh_theory_orig.size
    assert coh_real.shape[0] == coh_real_orig.size
    assert consider_all.sum() == coh_theory_orig.size
    assert consider_all.shape == coh_theory_orig.shape


@pytest.mark.parametrize("coh_threshold,expected_size", [(0.1, 15), (0.3, 9), (0.5, 3)])
def test_preproc_onestim_with_threshold(coh_threshold, expected_size):
    coh_real_orig = np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (3, 1))
    coh_theory_orig = coh_real_orig

    coh_theory, coh_real, consider_all = preproc_onestim(
        coh_theory_orig, coh_real_orig, coh_threshold
    )
    assert coh_theory.shape[0] == expected_size
    assert coh_real.shape[0] == expected_size
    assert consider_all.sum() == expected_size
    assert consider_all.shape == coh_theory_orig.shape


def test_preproc_twostim_no_threshold():
    coh_real_orig = np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (3, 5, 1))
    coh_theory_orig = coh_real_orig
    expected_size = 30  # after picking only triu(1)

    coh_theory, coh_real, consider_all = preproc_twostim(coh_theory_orig, coh_real_orig)
    assert coh_theory.shape[0] == expected_size
    assert coh_real.shape[0] == expected_size
    assert consider_all.sum() == expected_size
    assert consider_all.shape == coh_theory_orig.shape


@pytest.mark.parametrize(
    "coh_threshold,expected_size", [(0.1, 30), (0.3, 27), (0.5, 12)]
)
def test_preproc_twostim_with_threshold(coh_threshold, expected_size):
    coh_real_orig = np.tile([0.1, 0.2, 0.3, 0.4, 0.5], (3, 5, 1))
    coh_theory_orig = coh_real_orig

    coh_theory, coh_real, consider_all = preproc_twostim(
        coh_theory_orig, coh_real_orig, coh_threshold
    )
    assert coh_theory.shape[0] == expected_size
    assert coh_real.shape[0] == expected_size
    assert consider_all.sum() == expected_size
    assert consider_all.shape == coh_theory_orig.shape


def test_preproc_delta_all():
    pred_all = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    coh_real_all = np.array([3, 2, 1, 2, 2, 2, 1, 2, 3])
    consider_mask = np.full(pred_all.shape, True)

    expected_delta = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    pred_delta, coh_real_delta = preproc_delta(
        pred_all, coh_real_all, consider_mask, n_methods=3
    )
    assert np.allclose(pred_delta, 0.0)
    assert np.allclose(coh_real_delta, expected_delta.flatten())


def test_preproc_delta_masked():
    pred_all = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    coh_real_all = np.array([3, 2, 1, 2, 2, 2, 1, 2, 3])
    consider_mask = np.full(pred_all.shape, False)
    consider_mask[[1, 4, 7]] = True

    pred_delta, coh_real_delta = preproc_delta(
        pred_all, coh_real_all, consider_mask, n_methods=3
    )
    assert np.allclose(pred_delta, 0.0)
    assert np.allclose(coh_real_delta, 0.0)


def test_matrix_from_triu():
    result = matrix_from_triu(np.array([1, 2, 3]), 3)
    expected = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

    assert np.array_equal(result, expected)
