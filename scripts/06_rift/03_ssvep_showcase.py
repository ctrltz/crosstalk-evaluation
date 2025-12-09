import argparse
import mne
import numpy as np

from mne.minimum_norm import apply_inverse

from ctfeval.config import paths, params
from ctfeval.connectivity import rift_coherence
from ctfeval.datasets import get_rift_subject_ids, rift_subfolder
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.prepare import setup_inverse_operator


parser = argparse.ArgumentParser()
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)


def sensor_space(preproc_path, save_path, plot_info):
    # Process all subjects
    subject_ids = get_rift_subject_ids()
    coh_stim = np.zeros((len(subject_ids), len(plot_info.ch_names)))
    for i_subject, subject_id in enumerate(subject_ids):
        logger.info(f"Processing {subject_id}")

        # Load data
        epochs_path = preproc_path / f"{subject_id}_data_onestim-epo.fif"
        epochs = mne.read_epochs(epochs_path)
        stim = np.load(preproc_path / f"{subject_id}_data_onestim_tag1.npy")

        # Calculate brain-stimulus coherence
        coh_stim_full = rift_coherence(
            epochs.get_data(),
            stim,
            epochs.info["sfreq"],
            params.rift_fstim,
            kind="brain_stimulus",
            measure="coh",
        )

        # Pick common channels
        common, pick_idx, _ = np.intersect1d(
            epochs.info.ch_names, plot_info.ch_names, return_indices=True
        )
        assert len(common) == len(
            plot_info.ch_names
        ), "Expected all channels to be present"
        coh_stim[i_subject, :] = coh_stim_full[pick_idx]

        # Save the result
        np.save(save_path / f"{subject_id}_onestim_abscoh_stim1.npy", coh_stim_full)

    return coh_stim


def get_subject_ssvep(subject_id, labels, hm_path, preproc_path):
    # Head model
    fwd = mne.read_forward_solution(hm_path / f"{subject_id}-fwd.fif")
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    src = fwd["src"]

    # Get the average ERP
    epochs = mne.read_epochs(preproc_path / f"{subject_id}_data_onestim-epo.fif")
    evoked = epochs.average()

    # Get the ROI time courses
    inv = setup_inverse_operator(fwd, epochs.info)
    tcs = []
    for label in labels:
        stc = apply_inverse(evoked, inv, params.reg, method="eLORETA", label=label)
        label_tc = mne.extract_label_time_course(stc, label, src, mode="mean_flip")
        tcs.append(label_tc)

    return (evoked.times, *tcs)


def roi_ssvep_time_course(hm_path, preproc_path, v1_left, v1_right):
    times = None
    tcs_left = []
    tcs_right = []
    tcs_stim = []
    for subject_id in get_rift_subject_ids():
        times, tc_left, tc_right = get_subject_ssvep(
            subject_id, [v1_left, v1_right], hm_path, preproc_path
        )

        tcs_left.append(tc_left)
        tcs_right.append(tc_right)

        stim = np.load(preproc_path / f"{subject_id}_data_onestim_tag1.npy")
        tcs_stim.append(stim.mean(axis=0))

    tcs_left = np.stack(tcs_left)
    tcs_right = np.stack(tcs_right)
    tcs_stim = np.stack(tcs_stim)

    return times, tcs_left, tcs_right, tcs_stim


def main(tagging_type, random_phases):
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    hm_path = paths.rift / "headmodels"
    preproc_path = tag_folder / "preproc"
    sensor_path = tag_folder / "sensor_space"
    roi_path = tag_folder / "roi_space"
    sensor_path.mkdir(exist_ok=True)
    roi_path.mkdir(exist_ok=True)

    # Load channels that are common for all subjects
    plot_info = mne.io.read_info(paths.rift / "info" / "plot-info.fif")

    # Parcellation
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    v1_left, v1_right = p["pericalcarine-lh"], p["pericalcarine-rh"]

    # Get sensor-space patterns of brain-stimulus coherence as
    # reference to compare the leadfields to
    coh_stim = sensor_space(preproc_path, sensor_path, plot_info)
    coh_stim_ga = np.mean(coh_stim, axis=0)
    np.save(sensor_path / "grand_average_onestim_abscoh_stim1.npy", coh_stim_ga)

    # Get grand-average SSVEP time courses in V1 to show coherence
    times, tcs_left, tcs_right, tcs_stim = roi_ssvep_time_course(
        hm_path, preproc_path, v1_left, v1_right
    )

    # Save the results
    np.save(roi_path / "times.npy", times)
    np.save(roi_path / "stim.npy", np.squeeze(tcs_stim.mean(axis=0)))
    np.save(roi_path / "v1_left.npy", np.squeeze(tcs_left))
    np.save(roi_path / "v1_left_ga.npy", np.squeeze(tcs_left.mean(axis=0)))
    np.save(roi_path / "v1_right.npy", np.squeeze(tcs_right))
    np.save(roi_path / "v1_right_ga.npy", np.squeeze(tcs_right.mean(axis=0)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(int(args.tagging_type), int(args.random_phases))
