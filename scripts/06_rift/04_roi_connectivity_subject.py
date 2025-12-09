import argparse
import mne
import numpy as np

from mne.beamformer import make_lcmv, apply_lcmv_epochs
from mne.minimum_norm import apply_inverse_epochs

from ctfeval.config import paths, params, get_pipelines
from ctfeval.connectivity import rift_coherence
from ctfeval.datasets import rift_subfolder
from ctfeval.log import logger
from ctfeval.extraction import extract_label_time_course_centroid
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.prepare import setup_inverse_operator


parser = argparse.ArgumentParser()
parser.add_argument("--cond", required=True, choices=["onestim", "twostim"])
parser.add_argument("--subject", type=str, required=True)
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)


def process_condition(tagging_type, random_phases, subject_id, cond, p, pipelines):
    # Paths
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    hm_path = paths.rift / "headmodels"
    preproc_path = tag_folder / "preproc"
    brain_stim_path = tag_folder / "brain_stimulus"
    brain_brain_path = tag_folder / "brain_brain"

    brain_stim_path.mkdir(exist_ok=True)
    brain_brain_path.mkdir(exist_ok=True)

    # Load MEG data and the stimulation signal
    epochs_path = preproc_path / f"{subject_id}_data_{cond}-epo.fif"
    epochs = mne.read_epochs(epochs_path)
    sfreq = epochs.info["sfreq"]

    stim_path = preproc_path / f"{subject_id}_data_{cond}_tag1.npy"
    stim = np.load(stim_path)

    # Load the head model
    fwd = mne.read_forward_solution(hm_path / f"{subject_id}-fwd.fif")
    inv = setup_inverse_operator(fwd, epochs.info)

    # Loop over pipelines
    for pipeline in pipelines:
        inv_method, roi_method = pipeline
        logger.info(f"{inv_method} + {roi_method}")

        # Apply source reconstruction
        if inv_method == "eLORETA":
            stc = apply_inverse_epochs(epochs, inv, params.reg, method="eLORETA")
        elif inv_method == "LCMV":
            data_cov_lcmv = mne.compute_covariance(epochs, tmin=0, verbose=False)

            # Identity is used as the noise covariance matrix by default
            lcmv_filters = make_lcmv(
                epochs.info,
                fwd,
                data_cov_lcmv,
                pick_ori=None,
                weight_norm="unit-noise-gain",
                rank=None,
                verbose=False,
            )

            stc = apply_lcmv_epochs(epochs, lcmv_filters, verbose=False)

        # Get label time series
        src = fwd["src"]
        if roi_method == "centroid":
            label_tc = [
                np.array(
                    [
                        extract_label_time_course_centroid(
                            epoch_stc, label, src, "fsaverage", paths.subjects_dir
                        )[0]
                        for label in p.labels
                    ]
                )
                for epoch_stc in stc
            ]
        else:
            label_tc = mne.extract_label_time_course(
                stc, p.labels, src, mode=roi_method
            )
        logger.info(f"{len(label_tc)=}")
        logger.info(f"{label_tc[0].shape=}")

        del stc

        # Brain-stimulus coherence
        coh_stim = rift_coherence(
            label_tc,
            stim,
            sfreq,
            params.rift_fstim,
            kind="brain_stimulus",
            measure="coh",
        )
        np.save(
            brain_stim_path
            / f"{subject_id}_{cond}_{inv_method}_{roi_method}_abscoh_stim1.npy",
            coh_stim,
        )

        # Brain-brain abs(ImCoh)
        absimcoh = rift_coherence(
            label_tc,
            stim,
            sfreq,
            params.rift_fstim,
            kind="brain_brain",
            measure="imcoh",
            measure_kwargs=dict(return_abs=True),
        )
        np.save(
            brain_brain_path
            / f"{subject_id}_{cond}_{inv_method}_{roi_method}_absimcoh.npy",
            absimcoh,
        )

    logger.info("[done]")


def main(cond, subject_id, tagging_type, random_phases):
    logger.info(f"Processing {subject_id}")

    pipelines = get_pipelines(params.include_data_dependent)
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    process_condition(tagging_type, random_phases, subject_id, cond, p, pipelines)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.cond, args.subject, int(args.tagging_type), int(args.random_phases))
