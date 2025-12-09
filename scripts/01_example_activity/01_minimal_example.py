import numpy as np

from roiextract.filter import SpatialFilter

from ctfeval.config import paths, params
from ctfeval.ctf import get_ctf_source_cov
from ctfeval.io import dict_to_npz, init_subject
from ctfeval.metrics import estimate_correlation
from ctfeval.prepare import prepare_filter
from ctfeval.simulations import simulation_activity_minimal
from ctfeval.utils import vertno_to_index


def main():
    # Load the head model
    subject = "fsaverage"
    fwd, inv, lemon_info, p = init_subject(parcellation="DK")
    src = fwd["src"]
    target_label = p["postcentral-lh"]

    # Prepare the extraction methods
    inv_method = "eLORETA"
    roi_methods = ["mean", "mean_flip", "centroid"]

    # Select target sources
    targets = [
        2045,  # outside of the ROI
        34676,  # within the ROI, closer to hand/foot area
        100007,  # within the ROI, closer to face area
    ]
    target_indices = [vertno_to_index(src, "lh", t) for t in targets]
    n_targets = len(targets)
    unequal_std = np.array([1, 1, params.expA_unequal_std])

    source_std = np.zeros(fwd["nsource"])
    source_std[target_indices] = unequal_std
    source_cov = np.diag(source_std**2)

    # Estimate the CTF for each method
    sf_dict = {}
    ctf_equal = {}
    ctf_unequal = {}
    for roi_method in roi_methods:
        sf = SpatialFilter.from_inverse(
            fwd,
            inv,
            target_label,
            inv_method,
            params.reg,
            roi_method,
            subject,
            paths.subjects_dir,
        )
        ctf_eq = sf.get_ctf_fwd(fwd, mode="power", normalize="max")
        ctf_uneq = sf.get_ctf_fwd(fwd, mode="amplitude", normalize=None)
        ctf_uneq.data = get_ctf_source_cov(ctf_uneq.data, source_cov)

        sf.w = sf.w / np.abs(sf.w).max()
        sf_dict[roi_method] = sf
        ctf_equal[roi_method] = ctf_eq
        ctf_unequal[roi_method] = ctf_uneq

    # Extract theoretical estimates for each source
    corr_theory_equal = {}
    corr_theory_unequal = {}
    for roi_method in roi_methods:
        for ctf, corr in zip(
            [ctf_equal, ctf_unequal], [corr_theory_equal, corr_theory_unequal]
        ):
            ctf_method = np.squeeze(ctf[roi_method].data)
            corr[roi_method] = ctf_method[target_indices]

    # Simulate data
    sc_equal, raw_equal = simulation_activity_minimal(
        fwd, lemon_info, targets, std=1, random_state=params.seed
    )
    sc_unequal, raw_unequal = simulation_activity_minimal(
        fwd, lemon_info, targets, std=unequal_std, random_state=params.seed
    )

    # Estimate the reconstruction quality in simulations
    b, a = prepare_filter(sc_equal.sfreq, fmin=params.fmin, fmax=params.fmax)

    corr_sim_equal = {}
    corr_sim_unequal = {}
    grid = zip(
        [raw_equal, raw_unequal],
        [sc_equal, sc_unequal],
        [corr_sim_equal, corr_sim_unequal],
    )
    for raw, sc, corr in grid:
        for roi_method in roi_methods:
            corr[roi_method] = np.zeros((n_targets,))
            label_tc = sf_dict[roi_method].apply_raw(raw)
            for i, t in enumerate(targets):
                gt = sc[str(t)].waveform
                corr[roi_method][i] = estimate_correlation(label_tc, gt, b, a)

    # Save the results
    results = dict(
        roi_methods=roi_methods,
        targets=targets,
        target_roi=target_label.name,
    )
    results.update(dict_to_npz(sf_dict=sf_dict, transform=lambda sf: sf.w))
    results.update(
        dict_to_npz(ctf=ctf_equal, transform=lambda ctf: np.squeeze(ctf.data))
    )
    results.update(
        dict_to_npz(
            corr_theory_equal=corr_theory_equal,
            corr_theory_unequal=corr_theory_unequal,
            corr_sim_equal=corr_sim_equal,
            corr_sim_unequal=corr_sim_unequal,
        )
    )
    np.savez(paths.examples / "activity.npz", **results)


if __name__ == "__main__":
    main()
