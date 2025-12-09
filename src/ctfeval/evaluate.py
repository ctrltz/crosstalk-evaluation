import mne
import numpy as np

from numpy.linalg import norm
from numpy.polynomial import Polynomial
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from roiextract.utils import get_label_mask

from ctfeval.config import paths
from ctfeval.ctf import ctf_ratio_cov
from ctfeval.extraction import (
    apply_inverse_with_weights,
    extract_label_time_course_with_weights,
    get_filter,
)
from ctfeval.io import load_source_locations
from ctfeval.log import logger
from ctfeval.metrics import METRICS


def evaluate_sim(fwd, inv, raw, gt, label, pipelines, metrics, b, a):
    """
    Evaluate the extraction quality of provided pipelines in one simulation.
    """
    src = fwd["src"]

    # Save the source time courses to speed up computation
    stc = dict()
    W = dict()

    results_sim = {m: np.full(len(pipelines), np.nan) for m in METRICS}
    filters_sim = np.zeros((len(pipelines), fwd["nchan"]))

    logger.info(f"Evaluating {len(pipelines)} pipelines")
    for i_pipeline, pipeline in enumerate(pipelines):
        inv_method, reg, roi_method = pipeline
        logger.info(f"{inv_method=} | {reg=} | {roi_method=}")

        # Apply the inverse model
        if (inv_method, reg) not in stc:
            stc[(inv_method, reg)] = apply_inverse_with_weights(
                raw, fwd, inv, inv_method, label=label, reg=reg
            )

        if (inv_method, reg) not in W:
            stc_full, W[(inv_method, reg)], _ = apply_inverse_with_weights(
                raw, fwd, inv, inv_method, reg=reg, return_matrix=True
            )
            del stc_full

        # Extract the ROI time courses
        rec, weights = extract_label_time_course_with_weights(
            stc[(inv_method, reg)],
            label,
            src,
            roi_method,
            "fsaverage",
            paths.subjects_dir,
            return_weights=True,
        )

        # Construct the equivalent filter
        sf = get_filter(
            label,
            src,
            W[(inv_method, reg)],
            weights,
            inv_method,
            roi_method,
            method_params=dict(),
            ch_names=fwd["info"]["ch_names"],
        )
        filters_sim[i_pipeline, :] = np.squeeze(sf.w)

        # Sanity check: time series obtained with MNE methods and with the
        # spatial filter should match, at least up to a scaling constant
        sf_tc = sf.apply_raw(raw)
        dotprod = rec @ sf_tc.T / (norm(rec) * norm(sf_tc))
        assert np.allclose(dotprod, 1.0), f"{inv_method}_{roi_method}: {dotprod}"

        # Evaluate
        for m in metrics:
            results_sim[m][i_pipeline] = METRICS[m](rec, gt, b, a)

    return results_sim, filters_sim


def prepare_eval_labels(mask_mode, labels, simulation_path, simulation_id, src):
    if mask_mode == "roi":
        return labels

    label_names = [label.name for label in labels]
    source_labels = load_source_locations(
        simulation_path, simulation_id, src, label_names
    )
    source_labels = list(source_labels.values())

    return source_labels


def prepare_source_cov(source_cov_mode, simulation_path, simulation_id):
    source_cov = None

    if source_cov_mode == "full_info":
        cs_source = load_npz(simulation_path / f"{simulation_id}_cs_source.npz")
        source_cov = np.real(cs_source)

    return source_cov


def prepare_data_cov(data_cov_mode, simulation_path, simulation_id):
    if data_cov_mode == "auto":
        return "auto"

    raw = mne.io.read_raw(simulation_path / f"{simulation_id}_eeg.fif")
    data_cov = mne.compute_raw_covariance(raw, tstep=2.0).data
    return data_cov


def evaluate_theory(
    filters,
    pipelines,
    p,
    fwd,
    mask_mode,
    source_cov_mode,
    data_cov_mode,
    simulation_path,
    simulation_id,
):
    assert mask_mode in ["roi", "source"]
    assert source_cov_mode in ["no_info", "full_info"]
    assert data_cov_mode in ["auto", "actual"]

    src = fwd["src"]
    L = fwd["sol"]["data"]

    # Resolve all parameters
    eval_labels = prepare_eval_labels(
        mask_mode, p.labels, simulation_path, simulation_id, src
    )
    source_cov = prepare_source_cov(source_cov_mode, simulation_path, simulation_id)
    data_cov = prepare_data_cov(data_cov_mode, simulation_path, simulation_id)

    # NOTE: compute the large matrix only once per simulation and use as a
    # default value, this saves a lot of processing time
    data_cov_default = L @ source_cov @ L.T if source_cov is not None else L @ L.T
    # NOTE: adjust the scaling of the data covariance matrix to make all results comparable,
    # does not affect the ratio but allows plotting on the same scale
    if data_cov_mode == "actual":
        data_cov *= np.trace(data_cov_default) / np.trace(data_cov)

    ratio_sim = np.zeros((len(pipelines), p.n_labels))
    for i_pipeline, _ in enumerate(pipelines):
        for i_label, label in enumerate(eval_labels):
            mask = get_label_mask(label, src)
            w = np.atleast_2d(np.squeeze(filters[i_pipeline, i_label, :]))
            rat = ctf_ratio_cov(w, L, mask, data_cov_default, source_cov, data_cov)
            ratio_sim[i_pipeline, i_label] = rat

    return ratio_sim


def get_correlation_across_rois(val_theory, val_sim):
    rs = []
    betas = []
    for rec_theory, rec_simulation in zip(val_theory, val_sim):
        r = pearsonr(rec_theory, rec_simulation).statistic
        beta = Polynomial.fit(rec_theory, rec_simulation, deg=1).convert().coef[-1]
        rs.append(r)
        betas.append(beta)

    return rs, betas
