import argparse
import numpy as np

from ctfeval.config import paths, params, get_pipelines
from ctfeval.datasets import get_rift_subject_ids, rift_subfolder
from ctfeval.parcellations import PARCELLATIONS


parser = argparse.ArgumentParser()
parser.add_argument("--cond", required=True, choices=["onestim", "twostim"])
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)


def main(cond, tagging_type, random_phases):
    # Paths
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    brain_stim_path = tag_folder / "brain_stimulus"
    brain_brain_path = tag_folder / "brain_brain"

    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    subject_ids = get_rift_subject_ids()
    pipelines = get_pipelines(params.include_data_dependent)

    n_subjects = len(subject_ids)
    n_pipelines = len(pipelines)

    # Initialize arrays
    absimcoh = np.zeros((n_pipelines, n_subjects, p.n_labels, p.n_labels))
    coh_stim = np.zeros((n_pipelines, n_subjects, p.n_labels))

    # Main loop
    for i_pipe, (inv_method, roi_method) in enumerate(pipelines):
        for i_subject, subject_id in enumerate(subject_ids):
            # Brain-stimulus coherence
            coh_stim_file = (
                f"{subject_id}_{cond}_{inv_method}_{roi_method}_abscoh_stim1.npy"
            )
            coh_stim[i_pipe, i_subject, :] = np.load(brain_stim_path / coh_stim_file)

            # Brain-brain abs(ImCoh)
            absimcoh_file = (
                f"{subject_id}_{cond}_{inv_method}_{roi_method}_absimcoh.npy"
            )
            absimcoh[i_pipe, i_subject, :, :] = np.load(
                brain_brain_path / absimcoh_file
            )

    # Average over subjects
    avg_absimcoh = np.mean(absimcoh, axis=1)
    avg_coh_stim = np.mean(coh_stim, axis=1)

    # Save the results
    np.save(brain_stim_path / f"{cond}_avg_abscoh_stim.npy", avg_coh_stim)
    np.save(brain_stim_path / f"{cond}_abscoh_stim.npy", coh_stim)
    np.save(brain_brain_path / f"{cond}_avg_absimcoh.npy", avg_absimcoh)
    np.save(brain_brain_path / f"{cond}_absimcoh.npy", absimcoh)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.cond, int(args.tagging_type), int(args.random_phases))
