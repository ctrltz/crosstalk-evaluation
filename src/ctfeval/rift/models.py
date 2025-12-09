import numpy as np

from ctfeval.utils import vertno_to_index


class BaseSpreadModel:
    """
    Base class for the models of remaining field spread for RIFT data.
    """

    def __init__(self, labels, lh_vertno, rh_vertno):
        """
        Each model accepts positions (vertno) of SSVEP generators in both
        hemispheres as well as the list of ROIs to generate the predictions for.
        """
        self.labels = labels
        self.lh_vertno = lh_vertno
        self.rh_vertno = rh_vertno

        self.n_labels = len(labels)


class NoSpreadModel(BaseSpreadModel):
    """
    This model assumes no remaining field spread:

     * onestim case: brain-stimulus coherence will only be present in ROIs
     that contain either one of the SSVEP generators (hence or is used in the
     calculations)

     * twostim case: brain-brain ImCoh will be present for ROI pairs that
     capture both generators (hence and is used in the calculations, then the
     result is made symmetric)
    """

    def __init__(self, labels, lh_vertno, rh_vertno):
        super(NoSpreadModel, self).__init__(labels, lh_vertno, rh_vertno)
        self._precompute()

    def _precompute(self):
        """
        For each ROI, precompute whether it contains any of the SSVEP generators.
        """
        self.contains_lh = np.zeros(self.n_labels)
        self.contains_rh = np.zeros(self.n_labels)

        for idx, label in enumerate(self.labels):
            if self.lh_vertno is not None and label.hemi == "lh":
                self.contains_lh[idx] = self.lh_vertno in label.vertices
            if self.rh_vertno is not None and label.hemi == "rh":
                self.contains_rh[idx] = self.rh_vertno in label.vertices

    def pred_onestim(self):
        return np.logical_or(self.contains_lh, self.contains_rh)

    def pred_twostim(self):
        imcoh = np.logical_and(
            self.contains_lh[:, np.newaxis], self.contains_rh[np.newaxis, :]
        )
        return np.logical_or(imcoh, imcoh.T)


class DistanceSpreadModel(BaseSpreadModel):
    """
    This model assumes that the RFS is only based on the distance: both
    brain-stimulus coherence and brain-brain ImCoh should decay with higher
    distance to the generators of the SSVEP.
    """

    def __init__(self, labels, src, lh_vertno, rh_vertno):
        super(DistanceSpreadModel, self).__init__(labels, lh_vertno, rh_vertno)
        self.src = src
        self._precompute()

    def _precompute(self):
        """
        For each ROI, precompute the average distance from its vertices to the
        SSVEP generators.
        """
        lh_pos = (
            self.src[0]["rr"][self.lh_vertno, :] if self.lh_vertno is not None else None
        )
        rh_pos = (
            self.src[1]["rr"][self.rh_vertno, :] if self.lh_vertno is not None else None
        )

        self.dist_to_left = np.zeros(self.n_labels)
        self.dist_to_right = np.zeros(self.n_labels)
        for idx, label in enumerate(self.labels):
            label_pos = label.copy().restrict(self.src).pos
            self.dist_to_left[idx] = np.sqrt(
                np.sum((label_pos - lh_pos) ** 2, axis=1)
            ).mean()
            self.dist_to_right[idx] = np.sqrt(
                np.sum((label_pos - rh_pos) ** 2, axis=1)
            ).mean()

    def pred_onestim(self):
        n_ave = int(self.lh_vertno is not None) + int(self.rh_vertno is not None)
        return (self.dist_to_left + self.dist_to_right) / n_ave

    def pred_twostim(self):
        # Take the smallest of pairwise distances between target ROIs and source locations
        return np.minimum(
            self.dist_to_left[:, np.newaxis] + self.dist_to_right[np.newaxis, :],
            self.dist_to_right[:, np.newaxis] + self.dist_to_left[np.newaxis, :],
        )


class CTFSpreadModel(BaseSpreadModel):
    """
    This model uses CTF to estimate the effect of RFS on brain-stimulus
    coherence and brain-brain ImCoh.
    """

    def __init__(self, labels, src, lh_vertno, rh_vertno, gamma):
        super(CTFSpreadModel, self).__init__(labels, lh_vertno, rh_vertno)
        self.src = src
        self.gamma = gamma

    def pred_onestim(self, ctfs, approach="fast"):
        n_filters, _ = ctfs.shape
        source_cov, lh_idx, rh_idx = rift_source_cov(
            self.src,
            lh_vertno=self.lh_vertno,
            rh_vertno=self.rh_vertno,
            gamma=self.gamma,
            onestim=True,
        )

        ctfs_norm = normalize_ctf(ctfs, source_cov, lh_idx, rh_idx, approach)

        coh_theory = np.zeros((n_filters,))
        if self.lh_vertno is not None:
            coh_theory += ctfs_norm[:, lh_idx]
        if self.rh_vertno is not None:
            coh_theory += ctfs_norm[:, rh_idx]
        coh_theory = np.abs(coh_theory)

        return coh_theory

    def pred_twostim(self, ctfs, approach="fast"):
        n_filters, _ = ctfs.shape
        source_cov, lh_idx, rh_idx = rift_source_cov(
            self.src,
            lh_vertno=self.lh_vertno,
            rh_vertno=self.rh_vertno,
            gamma=self.gamma,
            onestim=False,
        )

        ctfs_norm = normalize_ctf(ctfs, source_cov, lh_idx, rh_idx, approach)
        ctf_left = np.squeeze(ctfs_norm[:, lh_idx])
        ctf_right = np.squeeze(ctfs_norm[:, rh_idx])

        ctf_product = ctf_left[:, np.newaxis] @ ctf_right[np.newaxis, :]
        imcoh_theory = np.abs(ctf_product - ctf_product.T)

        return imcoh_theory


def rift_source_cov(src, lh_vertno=None, rh_vertno=None, gamma=1, onestim=True):
    """
    Assumed source covariance matrix for both conditions (1/2 stimuli) of RIFT data.
    Gamma controls the source-space SNR (ratio between the variances of SSVEP sources
    and background activity).
    """
    lh_idx = None
    rh_idx = None
    assert onestim or (
        lh_vertno is not None and rh_vertno is not None
    ), "Expected both lh and rh vertices for twostim condition"

    n_sources = sum([s["nuse"] for s in src])
    source_var = np.ones((n_sources,)) / gamma
    if lh_vertno is not None:
        lh_idx = vertno_to_index(src, "lh", lh_vertno)
        source_var[lh_idx] = 1.0
    if rh_vertno is not None:
        rh_idx = vertno_to_index(src, "rh", rh_vertno)
        source_var[rh_idx] = 1.0

    source_cov = np.diag(source_var)

    # NOTE: here we adapt the source covariance to the experimental paradigm:
    #  * one stimulus - we assume identical activity (correlation = 1) in both hemispheres
    #  * two stimuli - we assume a pi / 2 phase shift (zero covariance) between hemispheres
    if onestim and lh_vertno is not None and rh_vertno is not None:
        source_cov[lh_idx, rh_idx] = source_cov[rh_idx, lh_idx] = 1.0

    return source_cov, lh_idx, rh_idx


def normalize_ctf(ctfs, source_cov, lh_idx, rh_idx, approach):
    if approach == "fast":
        # NOTE: to speed up computation, the multiplication is split in two parts since the
        # source covariance matrix is very close to being diagonal:
        #  * first, diagonal values only
        #  * second (addon), off-diagonal values, we expect two non-zero values in [lh_idx, rh_idx] and [rh_idx, lh_idx]
        ctf_norm = ctfs * np.diag(source_cov) @ ctfs.T
        if lh_idx is not None and rh_idx is not None:
            addon = np.zeros_like(ctfs)
            addon[:, lh_idx] = ctfs[:, rh_idx] * source_cov[rh_idx, lh_idx]
            addon[:, rh_idx] = ctfs[:, lh_idx] * source_cov[lh_idx, rh_idx]
            ctf_norm += addon @ ctfs.T
        ctfs_normalized = ctfs / np.sqrt(np.diag(ctf_norm))[:, np.newaxis]
    elif approach == "matmul":
        ctf_norm = ctfs @ source_cov @ ctfs.T
        ctfs_normalized = ctfs / np.sqrt(np.diag(ctf_norm))[:, np.newaxis]
    elif approach == "loops":
        ctfs_normalized = np.zeros_like(ctfs)
        for i_filter, ctf in enumerate(ctfs):
            ctfs_normalized[i_filter, :] = ctf / np.sqrt(ctf @ source_cov @ ctf.T)
    else:
        raise ValueError(f"Unexpected approach: {approach}")

    return ctfs_normalized


def init_model(model, src, labels, lh_vertno, rh_vertno, gamma):
    assert model in ["no_leakage", "distance", "ctf"]
    if model == "no_leakage":
        return NoSpreadModel(labels, lh_vertno, rh_vertno)
    elif model == "distance":
        return DistanceSpreadModel(labels, src, lh_vertno, rh_vertno)
    elif model == "ctf":
        return CTFSpreadModel(labels, src, lh_vertno, rh_vertno, gamma)
    else:
        raise ValueError("unreachable")


def get_theory_brain_stimulus(
    ctfs, labels, src, model, lh_vertno, rh_vertno, gamma, average=False
):
    n_subjects, n_methods, n_labels, n_sources = ctfs.shape
    m = init_model(model, src, labels, lh_vertno, rh_vertno, gamma)

    if model != "ctf":
        coh_theory = np.tile(
            m.pred_onestim()[np.newaxis, np.newaxis, :], (n_subjects, n_methods, 1)
        )
    else:
        coh_theory = np.zeros((n_subjects, n_methods, n_labels))
        for i_subject, ctfs_subject in enumerate(ctfs):
            for i_method, ctfs_method in enumerate(ctfs_subject):
                coh_theory[i_subject, i_method, :] = m.pred_onestim(ctfs_method)

    if average:
        coh_theory = coh_theory.mean(axis=0)

    return coh_theory


def get_theory_brain_brain(
    ctfs, labels, src, model, lh_vertno, rh_vertno, gamma, average=False
):
    n_subjects, n_methods, n_labels, n_sources = ctfs.shape
    m = init_model(model, src, labels, lh_vertno, rh_vertno, gamma)

    if model != "ctf":
        imcoh_theory = np.tile(
            m.pred_twostim()[np.newaxis, np.newaxis, :, :],
            (n_subjects, n_methods, 1, 1),
        )
    else:
        imcoh_theory = np.zeros((n_subjects, n_methods, n_labels, n_labels))
        for i_subject, ctfs_subject in enumerate(ctfs):
            for i_method, ctfs_method in enumerate(ctfs_subject):
                imcoh_theory[i_subject, i_method, :, :] = m.pred_twostim(ctfs_method)

    if average:
        imcoh_theory = imcoh_theory.mean(axis=0)

    return imcoh_theory


def get_theory(kind, *args, **kwargs):
    assert kind in ["brain_stimulus", "brain_brain"]

    if kind == "brain_stimulus":
        return get_theory_brain_stimulus(*args, **kwargs)

    return get_theory_brain_brain(*args, **kwargs)
