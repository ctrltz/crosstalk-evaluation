import argparse
import numpy as np
import mne

from ctfeval.config import paths, params
from ctfeval.connectivity import estimate_noise_floor
from ctfeval.datasets import get_rift_subject_ids, rift_subfolder
from ctfeval.log import logger


parser = argparse.ArgumentParser()
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)
parser.add_argument("-n", default=params.rift_noise_simulations)
parser.add_argument("--measure", choices=["coh", "imcoh"], required=True)
parser.add_argument("--onestim", action="store_true")


def main(tagging_type, random_phases, n_simulations, measure, onestim):
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    preproc_path = tag_folder / "preproc"
    theory_path = tag_folder / "theory"
    theory_path.mkdir(exist_ok=True)

    assert (
        onestim or random_phases
    ), "Two stimuli condition is only available for random phases"
    cond = "onestim" if onestim else "twostim"
    if measure == "coh":
        conn_params = dict(measure="coh")
    else:
        conn_params = dict(measure="imcoh", return_abs=True)

    subject_ids = get_rift_subject_ids()
    n_subjects = len(subject_ids)

    conn_noise = np.zeros((n_subjects, n_simulations))
    durations = np.zeros(n_subjects)
    seeds = np.random.SeedSequence(params.seed).generate_state(n_subjects)
    for i, (subject_id, seed) in enumerate(zip(subject_ids, seeds)):
        epochs_path = preproc_path / f"{subject_id}_data_{cond}-epo.fif"
        epochs = mne.read_epochs(epochs_path)

        sfreq = epochs.info["sfreq"]
        epoch_length = epochs.times[-1] - epochs.times[0]
        duration = len(epochs) * epoch_length
        logger.info(f"{subject_id=}: {sfreq=}, {epoch_length=}, {duration=}")

        conn_noise[i, :] = estimate_noise_floor(
            sfreq=sfreq,
            duration=duration,
            fstim=params.rift_fstim,
            conn_params=conn_params,
            n_simulations=n_simulations,
            random_state=seed,
        )
        durations[i] = duration

    conn_noise = conn_noise.mean(axis=0)
    conn_threshold = np.percentile(conn_noise, params.rift_noise_percentile)
    logger.info(f"{conn_threshold=}")

    np.savez(
        theory_path / f"noise_floor_{measure}_{cond}.npz",
        conn_noise=conn_noise,
        conn_threshold=conn_threshold,
        durations=durations,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        int(args.tagging_type),
        int(args.random_phases),
        args.n,
        args.measure,
        args.onestim,
    )
