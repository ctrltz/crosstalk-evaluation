import mne
import numpy as np

from tqdm import tqdm

from roiextract.filter import SpatialFilter

from ctfeval.config import paths, params, get_inverse_methods, get_roi_methods
from ctfeval.datasets import get_rift_subject_ids
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.prepare import (
    prepare_forward,
    prepare_source_space,
    setup_inverse_operator,
)


def prepare_head_model(subject_id, src, hm_path, info_path, save=True):
    # Use the same mne.Info as in the epochs
    info = mne.io.read_info(info_path / f"{subject_id}-info.fif")

    fwd = prepare_forward(
        paths.subjects_dir, "fsaverage", src, info, meg=True, save=False
    )
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    inv = setup_inverse_operator(fwd, info)

    if save:
        mne.write_forward_solution(
            hm_path / f"{subject_id}-fwd.fif", fwd, overwrite=True
        )

    return fwd, inv


def prepare_filter_ctfs(
    subject_ids,
    p,
    inv_method,
    roi_methods,
    hm_path,
    info_path,
    reg=params.reg,
    save=True,
):
    src = prepare_source_space(
        paths.subjects_dir, "fsaverage", spacing="oct6", plot_src=False
    )

    n_subjects = len(subject_ids)
    n_methods = len(roi_methods)
    n_sources = sum([s["nuse"] for s in src])

    ctfs = np.zeros((n_subjects, n_methods, p.n_labels, n_sources))
    for i_subject, subject_id in enumerate(subject_ids):
        fwd, inv = prepare_head_model(subject_id, src, hm_path, info_path, save=save)

        for i_method, roi_method in enumerate(roi_methods):
            for i_label, label in enumerate(
                tqdm(p.labels, desc=f"{subject_id}, {inv_method} + {roi_method}")
            ):
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
                # NOTE: normalization of CTF is performed using model-specific source
                # covariance later, here no normalization is needed
                ctf = sf.get_ctf_fwd(fwd, mode="amplitude", normalize=None)
                ctfs[i_subject, i_method, i_label, :] = np.squeeze(ctf.data)

    return ctfs


def prepare_mean_leadfield(hm_path, info_plot, subject_ids):
    # Extract the lead field for common channels
    leadfields = []
    for subject_id in subject_ids:
        fwd = mne.read_forward_solution(hm_path / f"{subject_id}-fwd.fif")
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
        fwd_pick = fwd.pick_channels(info_plot.ch_names, ordered=True)
        leadfields.append(fwd_pick["sol"]["data"])

    leadfields_stacked = np.stack(leadfields, axis=0)
    leadfield_mean = leadfields_stacked.mean(axis=0)

    return leadfield_mean


def prepare_common_info(info_path, subject_ids):
    # Keep only channels which are present for each subject
    ch_names_overlap = None
    for subject_id in subject_ids:
        info = mne.io.read_info(info_path / f"{subject_id}-info.fif")
        meg_idx = mne.channel_indices_by_type(info)["mag"]
        info_mag = mne.pick_info(info, meg_idx)
        if ch_names_overlap is None:
            ch_names_overlap = set(info_mag.ch_names)
        else:
            ch_names_overlap &= set(info_mag.ch_names)
    ch_names_overlap = sorted(list(ch_names_overlap))

    # Pick the info from one subject, keeping only common channels
    info = mne.io.read_info(info_path / f"{subject_ids[0]}-info.fif")
    pick_idx = mne.pick_channels(
        info.ch_names, include=list(ch_names_overlap), ordered=True
    )
    info_plot = mne.pick_info(info, pick_idx)

    return info_plot


def main():
    # Paths
    hm_path = paths.rift / "headmodels"
    hm_path.mkdir(exist_ok=True)
    info_path = paths.rift / "info"
    ctf_file = paths.rift / "ctf_array.npy"

    # Subject IDs
    subject_ids = get_rift_subject_ids()

    # Get a mne.Info object with common channels
    info_plot = prepare_common_info(info_path, subject_ids)
    info_plot.save(info_path / "plot-info.fif")

    # Compute the spatial filters for all methods
    inv_method = get_inverse_methods(include_data_dependent=False)[0]
    roi_methods = get_roi_methods(include_data_dependent=False)

    # Prepare CTFs for DK parcellation
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    ctfs = prepare_filter_ctfs(
        subject_ids, p, inv_method, roi_methods, hm_path, info_path
    )
    np.save(ctf_file, ctfs)

    # Prepare an average leadfield for plotting
    leadfield_mean = prepare_mean_leadfield(hm_path, info_plot, subject_ids)
    np.save(hm_path / "leadfield_mean.npy", leadfield_mean)


if __name__ == "__main__":
    main()
