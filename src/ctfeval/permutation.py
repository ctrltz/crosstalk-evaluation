import numpy as np


def permute(data, sfreq, seg_len=2, random_state=None):
    """
    Split the data into segments of seg_len seconds and shuffle the segments.
    The implementation is borrowed from Jamshidi Idaji et al. (2022)
    """

    # Number of channels and samples
    nchan, n_samples = data.shape

    # Sample number to segment the data
    seg_len = int(seg_len * sfreq)

    # Divide the data
    n_seg = int(n_samples // seg_len)
    n_omit = int(n_samples % seg_len)

    data_rest = data[:, -n_omit:] if n_omit > 0 else np.empty((nchan, 0))
    data_truncated = data[:, :-n_omit] if n_omit > 0 else data
    data_truncated = data_truncated.reshape((nchan, seg_len, n_seg), order="F")

    # Permute
    data_perm = np.zeros([nchan, n_samples])
    rng = np.random.default_rng(seed=random_state)
    for ch in range(nchan):
        perm = rng.permutation(n_seg)
        ts_perm = data_truncated[ch, :, perm]
        ts_perm = ts_perm.reshape([1, (n_samples - n_omit)])
        data_perm[ch, :] = np.concatenate((ts_perm[0, :], data_rest[ch, :]), axis=0)

    return data_perm


# NOTE: ica_unmix and ica_mix are based on the source code of MNE
# They are checked for each subject by unmixing and mixing back the
# raw data and comparing with the original
def ica_unmix(raw, ica):
    return ica.get_sources(raw).get_data()


def ica_mix(components, ica):
    # Extracted from ica._apply
    data = ica.pca_components_[: ica.n_components_].T @ ica.mixing_matrix_ @ components
    if ica.pca_mean_ is not None:
        data += ica.pca_mean_[:, None]
    if ica.noise_cov is None:
        data *= ica.pre_whitener_
    else:
        raise RuntimeError("did not expect to end up here")

    return data


def permute_ica(raw, ica, seg_len=2, random_state=None):
    components = ica_unmix(raw, ica)
    components_perm = permute(
        components, raw.info["sfreq"], seg_len=seg_len, random_state=random_state
    )
    eeg_perm = ica_mix(components_perm, ica)

    raw_perm = raw.copy()
    raw_perm._data = eeg_perm

    return raw_perm
