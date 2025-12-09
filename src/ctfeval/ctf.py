import numpy as np

from roiextract.filter import SpatialFilter
from roiextract.optimize import ctf_optimize_label
from roiextract.quantify import ctf_quantify_label

from ctfeval.config import paths


def ctf_ratio_cov(w, L, mask, data_cov_default, source_cov, data_cov):
    data_cov_auto = isinstance(data_cov, str) and data_cov == "auto"
    if data_cov_auto:
        assert data_cov_default is not None
        data_cov = data_cov_default
    denom = w @ data_cov @ w.T

    if source_cov is not None:
        nom = w @ L[:, mask] @ source_cov[np.ix_(mask, mask)] @ L[:, mask].T @ w.T
    else:
        nom = w @ L[:, mask] @ L[:, mask].T @ w.T

    result = np.squeeze(nom / denom)

    if np.ndim(result) > 0:
        return np.diag(result)

    return result


def get_max_ctf_ratios(fwd, p):
    ratios = np.zeros(p.n_labels)
    for i, label in enumerate(p.labels):
        _, props = ctf_optimize_label(fwd, label, "mean_flip", lambda_=0, quantify=True)
        ratios[i] = props["rat"]

    return ratios


def get_achieved_ctf_ratios(fwd, p, inv, inv_method, roi_method, reg=0.05):
    ratios = np.zeros((p.n_labels,))
    for i, label in enumerate(p.labels):
        sf = SpatialFilter.from_inverse(
            fwd,
            inv,
            label,
            inv_method,
            reg,
            roi_method,
            "fsaverage",
            paths.subjects_dir,
        )
        props = ctf_quantify_label(sf.w, fwd, label)
        ratios[i] = props["rat"]

    return ratios


def get_ctf_source_cov(ctf, source_cov):
    # NOTE: covers only power mode for uncorrelated sources, assuming (vert,) shape
    nom = (ctf**2) * np.diag(source_cov)[:, np.newaxis]
    denom = ctf.T @ source_cov @ ctf
    return nom / denom


def spurious_coherence(ctf1, ctf2, source_cov=None):
    if source_cov is not None:
        nom = (ctf1 * np.diag(source_cov)) @ ctf2.T
        denom = np.sqrt((ctf1 @ source_cov @ ctf1.T) * (ctf2 @ source_cov @ ctf2.T))
    else:
        nom = ctf1 @ ctf2.T
        denom = np.sqrt((ctf1 @ ctf1.T) * (ctf2 @ ctf2.T))

    return np.abs(nom / denom)
