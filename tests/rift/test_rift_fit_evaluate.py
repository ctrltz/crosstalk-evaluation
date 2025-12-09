import numpy as np
import pytest

from ctfeval.config import paths
from ctfeval.io import load_source_space
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.rift.fit_evaluate import define_search_grid, fit_and_predict


@pytest.mark.parametrize(
    "lh_label,rh_label,gammas,model",
    [
        ("pericalcarine-lh", "pericalcarine-rh", [1, 2, 3], "no_leakage"),
        ("pericalcarine-lh", "pericalcarine-rh", [1, 2, 3], "distance"),
        ("pericalcarine-lh", "pericalcarine-rh", [1, 2, 3], "ctf"),
        ("precentral-lh", "precentral-rh", [1, 2, 3, 4, 5], "no_leakage"),
        ("precentral-lh", "precentral-rh", [1, 2, 3, 4, 5], "distance"),
        ("precentral-lh", "precentral-rh", [1, 2, 3, 4, 5], "ctf"),
    ],
)
def test_define_search_grid(lh_label, rh_label, gammas, model):
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)

    grid = define_search_grid(
        src,
        p[lh_label],
        p[rh_label],
        gammas,
        model,
    )

    lh_size = len(p[lh_label].restrict(src).vertices)
    rh_size = len(p[rh_label].restrict(src).vertices)
    n_gammas = len(gammas)
    expected_count = lh_size * rh_size
    if model == "ctf":
        expected_count *= n_gammas
    assert len(grid) == expected_count


@pytest.mark.parametrize("model", ["no_leakage", "ctf"])
def test_fit_and_predict_linear(model):
    theory_fit = np.arange(5)
    real_fit = np.arange(5)
    theory_all = np.arange(10)

    fm, intercept, slope, pred_fit, pred_all = fit_and_predict(
        theory_fit, real_fit, theory_all, model
    )

    assert np.isclose(intercept, 0)
    assert np.isclose(slope, 1)
    assert np.allclose(pred_fit, real_fit)
    assert np.allclose(pred_all, theory_all)


@pytest.mark.parametrize("gt_slope", [1, 2])
def test_fit_and_predict_power_law(gt_slope):
    theory_fit = np.arange(1, 5)
    real_fit = 1.0 / theory_fit**gt_slope
    theory_all = np.arange(1, 10)

    fm, intercept, slope, pred_fit, pred_all = fit_and_predict(
        theory_fit, real_fit, theory_all, model="distance"
    )

    assert np.isclose(intercept, 1)
    assert np.isclose(slope, -gt_slope)
    assert np.allclose(pred_fit, real_fit)
    assert np.allclose(pred_all, 1.0 / theory_all**gt_slope)
