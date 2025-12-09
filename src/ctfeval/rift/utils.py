import numpy as np


def preproc_onestim(coh_theory, coh_real, coh_threshold=None):
    """
    For onestim, consider only values which exceed the coherence threshold during model fit.
    """
    if coh_threshold is not None:
        # Consider only significant connectivity values
        consider_mask = coh_real >= coh_threshold
    else:
        consider_mask = np.full(coh_real.shape, True)

    coh_theory = coh_theory.copy()[consider_mask]
    coh_real = coh_real.copy()[consider_mask]

    return coh_theory, coh_real, consider_mask


def preproc_twostim(coh_theory, coh_real, coh_threshold=None):
    """
    For twostim, consider only upper-triangular edges with connectivity values
    which exceed the coherence threshold during model fit.
    """
    # Pick only the upper triangular part
    n_methods = coh_theory.shape[0]
    triu_part = np.triu(np.ones_like(coh_theory[0, :, :]), 1) > 0
    consider_mask = np.tile(triu_part[np.newaxis, :, :], (n_methods, 1, 1))

    if coh_threshold is not None:
        # Consider only significant connectivity values
        consider_mask = np.logical_and(consider_mask, coh_real >= coh_threshold)

    coh_theory = coh_theory.copy()[consider_mask]
    coh_real = coh_real.copy()[consider_mask]

    return coh_theory, coh_real, consider_mask


def preproc_delta(pred_all, coh_real, consider_mask, n_methods):
    # Reshape to get 1 row per method
    # NOTE: use the same approach for all arrays to ensure consistency
    pred_all = pred_all.copy().reshape((n_methods, -1))
    coh_real = coh_real.copy().reshape((n_methods, -1))
    consider_mask = consider_mask.copy().reshape((n_methods, -1))

    # Get and subtract the mean over pipelines
    pred_mean = pred_all.mean(axis=0)[np.newaxis, :]
    coh_real_mean = coh_real.mean(axis=0)[np.newaxis, :]

    pred_delta = pred_all - np.tile(pred_mean, (n_methods, 1))
    coh_real_delta = coh_real - np.tile(coh_real_mean, (n_methods, 1))

    return pred_delta[consider_mask], coh_real_delta[consider_mask]


def matrix_from_triu(coh_values, size):
    data = np.zeros((size, size))
    triu_indices = np.triu_indices_from(data, 1)
    data[triu_indices] = coh_values
    data = data + data.T

    return data
