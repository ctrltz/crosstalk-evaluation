import numpy as np

from roiextract.filter import SpatialFilter

from ctfeval.config import paths, params
from ctfeval.connectivity import imcoh
from ctfeval.extraction import (
    apply_inverse_with_weights,
    extract_label_time_course_with_weights,
)
from ctfeval.io import dict_to_npz, init_subject
from ctfeval.simulations import simulation_connectivity_minimal


def label_imcoh(raw, fwd, inv, label1, label2, inv_method, roi_method, reg=0.05):
    src = fwd["src"]

    tcs = []
    for label in [label1, label2]:
        stc = apply_inverse_with_weights(raw, fwd, inv, inv_method, reg, label)
        tc = extract_label_time_course_with_weights(
            stc,
            label,
            src,
            roi_method,
            subject="fsaverage",
            subjects_dir=paths.subjects_dir,
        )
        tcs.append(tc)
    tcs = np.vstack(tcs)

    return imcoh(tcs, raw.info["sfreq"])


def main():
    fwd, inv, lemon_info, p = init_subject(parcellation="DK")

    target = ["caudalanteriorcingulate-lh", "isthmuscingulate-rh"]
    interference = ["superiorfrontal-lh", "inferiorparietal-rh"]
    target_labels = [p[name] for name in target]
    interference_labels = [p[name] for name in interference]

    inv_method = "eLORETA"
    reg = 0.05
    roi_method = "mean_flip"

    # Prepare the CTFs
    ctf = {}
    for label in target_labels + interference_labels:
        sf = SpatialFilter.from_inverse(
            fwd,
            inv,
            label,
            inv_method,
            reg,
            roi_method,
            subject="fsaverage",
            subjects_dir=paths.subjects_dir,
        )
        ctf[label.name] = sf.get_ctf_fwd(fwd, mode="power", normalize="max")

    # Simulate all configurations
    raw_none = simulation_connectivity_minimal(
        fwd, lemon_info, target_labels, interference_labels, {}, params.seed
    )
    raw_target_only = simulation_connectivity_minimal(
        fwd,
        lemon_info,
        target_labels,
        interference_labels,
        {
            ("target-lh", "target-rh"): dict(
                phase_lag=params.conn_example_gt_lag, coh=params.conn_example_gt_coh
            )
        },
        params.seed,
    )
    raw_interference_only = simulation_connectivity_minimal(
        fwd,
        lemon_info,
        target_labels,
        interference_labels,
        {
            ("interference-lh", "interference-rh"): dict(
                phase_lag=-params.conn_example_gt_lag,
                coh=params.conn_example_gt_coh,
            )
        },
        params.seed,
    )
    sc_both, raw_both = simulation_connectivity_minimal(
        fwd,
        lemon_info,
        target_labels,
        interference_labels,
        {
            ("target-lh", "target-rh"): dict(
                phase_lag=params.conn_example_gt_lag, coh=params.conn_example_gt_coh
            ),
            ("interference-lh", "interference-rh"): dict(
                phase_lag=-params.conn_example_gt_lag,
                coh=params.conn_example_gt_coh,
            ),
        },
        params.seed,
        return_sc=True,
    )

    # Ground-truth ImCoh
    imcoh_t = {}
    imcoh_i = {}
    freqs, imcoh_t["gt"] = imcoh(
        np.vstack([sc_both["target-lh"].waveform, sc_both["target-rh"].waveform]),
        sc_both.sfreq,
    )
    freqs, imcoh_i["gt"] = imcoh(
        np.vstack(
            [sc_both["interference-lh"].waveform, sc_both["interference-rh"].waveform]
        ),
        sc_both.sfreq,
    )

    # Estimated ImCoh
    for case, raw in zip(
        ["none", "target", "interference", "both"],
        [raw_none, raw_target_only, raw_interference_only, raw_both],
    ):
        f, imcoh_t[case] = label_imcoh(
            raw, fwd, inv, *target_labels, "eLORETA", "mean_flip"
        )
        f, imcoh_i[case] = label_imcoh(
            raw, fwd, inv, *interference_labels, "eLORETA", "mean_flip"
        )

    # Save the results
    results = dict(target=target, interference=interference, freqs=freqs)
    results.update(dict_to_npz(ctf=ctf, transform=lambda ctf: np.squeeze(ctf.data)))
    results.update(dict_to_npz(imcoh_t=imcoh_t, imcoh_i=imcoh_i))
    np.savez(paths.examples / "connectivity.npz", **results)


if __name__ == "__main__":
    main()
