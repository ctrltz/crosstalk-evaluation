"""
Common approach for extraction of ROI activity = inverse modeling followed by ROI aggregation
"""

import mne
import numpy as np

from mne.beamformer import apply_lcmv, make_lcmv, apply_lcmv_raw
from mne.evoked import EvokedArray
from mne.minimum_norm import apply_inverse, prepare_inverse_operator

from roiextract.filter import SpatialFilter
from roiextract.utils import get_label_mask

from ctfeval.config import params


MNE_LABEL_METHODS = ["mean", "mean_flip", "pca_flip"]


def prepare_cov_lcmv(raw, duration=4.0, overlap=2.0):
    epochs = mne.make_fixed_length_epochs(
        raw, duration=duration, overlap=overlap, verbose=False
    )

    data_cov_lcmv = mne.compute_covariance(epochs, verbose=False)

    return data_cov_lcmv


def get_inverse_matrix(inv, info, inv_method):
    n_chans = len(info["ch_names"])
    evoked = EvokedArray(np.eye(n_chans), info)

    if inv_method == "LCMV":
        stc = apply_lcmv(evoked, inv, verbose=False)
    else:  # minimum norm
        stc = apply_inverse(
            evoked, inv, method=inv_method, prepared=True, verbose=False
        )

    return stc.data


def apply_inverse_with_weights(
    raw,
    fwd,
    inv,
    inv_method,
    reg=params.reg,
    label=None,
    cov_duration=4.0,
    cov_overlap=2.0,
    return_matrix=False,
):
    if inv_method in ["dSPM", "eLORETA"]:
        inv_op = prepare_inverse_operator(
            inv,
            nave=1,
            lambda2=reg,
            method=inv_method,
            method_params=None,
            copy=True,
            verbose=False,
        )
        stc = mne.minimum_norm.apply_inverse_raw(
            raw,
            inverse_operator=inv_op,
            method=inv_method,
            label=label,
            lambda2=reg,
            prepared=True,
            verbose=False,
        )
    elif inv_method == "LCMV":
        data_cov_lcmv = prepare_cov_lcmv(
            raw, duration=cov_duration, overlap=cov_overlap
        )

        # Identity is used as the noise covariance matrix by default
        inv_op = make_lcmv(
            raw.info,
            fwd,
            data_cov_lcmv,
            reg=reg,
            label=label,
            pick_ori=None,
            weight_norm="unit-noise-gain",
            rank=None,
            verbose=False,
        )

        stc = apply_lcmv_raw(raw, inv_op, verbose=False)
    else:
        raise ValueError(f"Unsupported inverse method: {inv_method}")

    if not return_matrix:
        return stc

    return stc, get_inverse_matrix(inv_op, raw.info, inv_method), inv_op


def get_aggregation_weights(stc, label, src, roi_method):
    """
    MNE approaches only, centroid is processed separately
    """
    from mne.fixes import _safe_svd
    from mne.label import label_sign_flip

    assert roi_method in MNE_LABEL_METHODS

    signflip = label_sign_flip(label, src)
    if roi_method == "mean_flip":
        return np.atleast_2d(signflip) / signflip.size
    if roi_method == "mean":
        return np.ones((1, signflip.size)) / signflip.size

    # only pca_flip from here
    U, _, _ = _safe_svd(stc.data, full_matrices=False)
    sign = np.sign(np.dot(U[:, 0], signflip))
    return np.atleast_2d(sign * U[:, 0])


def extract_label_time_course_centroid(
    stc, label, src, subject, subjects_dir, return_weights=False
):
    center_vertno = label.center_of_mass(
        subject=subject, restrict_vertices=src, subjects_dir=subjects_dir
    )
    hemi_idx = 1 if label.hemi == "rh" else 0
    offset = len(stc.vertices[0]) if label.hemi == "rh" else 0
    center_idx = offset + np.where(stc.vertices[hemi_idx] == center_vertno)[0]

    n_vertices = sum(v.size for v in stc.vertices)
    weights = np.zeros((n_vertices,))
    weights[center_idx] = 1

    if return_weights:
        return stc.data[center_idx, :], np.atleast_2d(weights)

    return stc.data[center_idx, :]


def extract_label_time_course_with_weights(
    stc, label, src, roi_method, subject=None, subjects_dir=None, return_weights=False
):
    if roi_method in MNE_LABEL_METHODS:
        weights = get_aggregation_weights(stc, label, src, roi_method)
        label_tc = mne.extract_label_time_course(
            stc, label, src, mode=roi_method, verbose=False
        )

        if return_weights:
            return label_tc, weights

        return label_tc

    if roi_method == "centroid":
        return extract_label_time_course_centroid(
            stc,
            label,
            src,
            subject=subject,
            subjects_dir=subjects_dir,
            return_weights=return_weights,
        )

    raise ValueError(
        f"Unsupported method for aggregation of ROI activity: {roi_method}"
    )


def get_filter(
    label, src, W, weights, inv_method, roi_method, method_params=None, ch_names=None
):
    mask = get_label_mask(label, src)
    w = weights @ W[mask, :]
    return SpatialFilter(
        np.atleast_2d(np.squeeze(w)),
        method=f"{inv_method}_{roi_method}",
        method_params=method_params,
        ch_names=ch_names,
        name=label.name,
    )
