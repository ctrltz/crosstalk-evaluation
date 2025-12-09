import numpy as np

from itertools import product
from sklearn.linear_model import LinearRegression

from ctfeval.rift.models import get_theory
from ctfeval.rift.utils import preproc_onestim, preproc_twostim, preproc_delta


def define_search_grid(
    src, lh_search_space, rh_search_space, gammas, model, hemi="both"
):
    lh_vertices = lh_search_space.restrict(src).vertices if hemi != "rh" else [None]
    rh_vertices = rh_search_space.restrict(src).vertices if hemi != "lh" else [None]
    gammas = gammas if model == "ctf" else [None]

    return list(product(lh_vertices, rh_vertices, gammas))


def fit_and_predict(
    theory_fit, real_fit, theory_all, model, pred=True, fit_intercept=True
):
    # Prepare the data vectors
    theory = np.atleast_2d(theory_fit.copy()).T
    theory_full = np.atleast_2d(theory_all.copy()).T
    real = np.atleast_2d(real_fit.copy()).T
    if model == "distance":
        # NOTE: for distance, we fit a power-law model to allow for arbitrary slope
        # of the dependency function
        # c = k * d ^ gamma <-> log(c) = log(k) + gamma * log(d)
        theory = np.log(theory)
        theory_full = np.log(theory_full)
        real = np.log(real)

    # Fit the model and extract the parameters
    fm = LinearRegression(fit_intercept=fit_intercept).fit(theory, real)
    slope = fm.coef_.item(0)
    intercept = fm.intercept_.item(0)
    if model == "distance":
        # convert intercept to linear scale to match the rest of the models
        intercept = np.exp(intercept)

    if not pred:
        return fm, intercept, slope

    # Get predictions
    pred_fit = np.squeeze(fm.predict(theory))
    pred_all = np.squeeze(fm.predict(theory_full))
    if model == "distance":
        # convert predictions to linear scale to match the rest of the models
        pred_fit = np.exp(pred_fit)
        pred_all = np.exp(pred_all)

    return fm, intercept, slope, pred_fit, pred_all


def fit_evaluate(
    ctfs,
    labels,
    model,
    kind,
    coh_real,
    coh_threshold,
    src,
    lh_vertno,
    rh_vertno,
    gamma,
    metrics,
    fit_intercept=True,
    delta_threshold=1e-6,
):
    assert model in ["no_leakage", "distance", "ctf"]
    assert kind in ["brain_stimulus", "brain_brain"]

    # Get theoretical predictions for the respective model
    coh_theory = get_theory(
        kind, ctfs, labels, src, model, lh_vertno, rh_vertno, gamma, average=True
    )
    n_methods = coh_theory.shape[0]

    # Consider only significant connectivity values during fit, flatten arrays
    preproc_fun = preproc_onestim if kind == "brain_stimulus" else preproc_twostim
    coh_theory_fit, coh_real_fit, consider_fit = preproc_fun(
        coh_theory, coh_real, coh_threshold
    )
    coh_theory_all, coh_real_all, consider_all = preproc_fun(coh_theory, coh_real)

    # Fit the model and get predictions for all data points
    fm, intercept, slope, pred_fit, pred_all = fit_and_predict(
        coh_theory_fit, coh_real_fit, coh_theory_all, model
    )

    # Evaluate the resulting fit, going through all metrics
    result = {
        "lh_seed": lh_vertno,
        "rh_seed": rh_vertno,
        "gamma": gamma,
        "intercept": intercept,
        "slope": slope,
    }
    for name, fun in metrics.items():
        # Option 1 (raw): evaluate based on the raw values for different methods
        result[f"{name}_raw_fit"] = fun(pred_fit, coh_real_fit)
        result[f"{name}_raw_all"] = fun(pred_all, coh_real_all)

        # Option 2 (delta): evaluate based on differences between values for one methods
        # and the mean across methods
        # NOTE: consider masks are flattened to match pred and coh_real
        pred_delta_fit, coh_real_delta_fit = preproc_delta(
            pred_all, coh_real_all, consider_fit[consider_all], n_methods
        )
        pred_delta_all, coh_real_delta_all = preproc_delta(
            pred_all, coh_real_all, consider_all[consider_all], n_methods
        )

        # NOTE: threshold is used below to prevent calculating correlation using values
        # on the order of eps (no_leakage and distance lead to the same values for
        # all methods and have no delta by construction)

        # Fit
        result[f"{name}_delta_fit"] = np.nan
        if np.any(np.abs(pred_delta_fit) > delta_threshold):
            result[f"{name}_delta_fit"] = fun(pred_delta_fit, coh_real_delta_fit)

        # All
        result[f"{name}_delta_all"] = np.nan
        if np.any(np.abs(pred_delta_all) > delta_threshold):
            result[f"{name}_delta_all"] = fun(pred_delta_all, coh_real_delta_all)

    return result
