import argparse
import numpy as np
import mne

from ctfeval.config import paths, params
from ctfeval.datasets import get_lemon_filename
from ctfeval.io import load_head_model, load_raw
from ctfeval.infer import (
    infer_sensor_space_snr,
    infer_source_power,
    infer_noise_level_cv,
    infer_noise_level_amplifier,
)
from ctfeval.log import logger
from ctfeval.prepare import setup_inverse_operator


parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, required=True)
parser.add_argument("--spacing", choices=["oct6", "ico4", "ico5"], default="oct6")
parser.add_argument("--tmax", type=int)
parser.add_argument("--all", action="store_true")
parser.add_argument("--sensor-space-snr", action="store_true")
parser.add_argument("--source-power", action="store_true")
parser.add_argument("--noise-level-cv", action="store_true")
parser.add_argument("--noise-level-amplifier", action="store_true")


def main(subject, spacing, args):
    # Check if preprocessed data exist for the provided subject ID
    preproc_path = paths.lemon_data / subject if params.lemon_bids else paths.lemon_data
    search_mask = f"{subject}_*.set"
    preproc_files = list(preproc_path.glob(search_mask))
    if not preproc_files:
        raise ValueError(f"Could not find subject {subject}")

    # Paths
    save_path = paths.derivatives / "real_data" / "infer_params"
    subject_save_path = save_path / subject
    subject_save_path.mkdir(parents=True, exist_ok=True)

    # Enable all steps if requested
    estimate_sensor_space_snr = args.all or args.sensor_space_snr
    estimate_source_power = args.all or args.source_power
    estimate_noise_level_cv = args.all or args.noise_level_cv
    estimate_noise_level_amplifier = args.all or args.noise_level_amplifier

    # Raw data is used only for inferring the level of amplifier noise
    raw_path = paths.lemon_raw_data / subject / "RSEEG" / f"{subject}.vhdr"
    logger.info(f"Checking whether raw EEG file exists: {raw_path.exists()}")
    if estimate_noise_level_amplifier and raw_path.exists():
        logger.info("Inferring the level of amplifier noise")
        raw_full = mne.io.read_raw_brainvision(raw_path)
        noise_level, amp_noise, spec_avg, freqs_raw = infer_noise_level_amplifier(
            raw_full, fmin=params.fmin, fmax=params.fmax, famp=params.infer_famp
        )
        np.savez(
            subject_save_path / "amplifier_noise.npz",
            noise_level=noise_level,
            amp_noise=amp_noise,
            spec_avg=spec_avg,
        )
        np.save(save_path / "freqs_raw.npy", freqs_raw)

    # Preprocessed data is used to inferred SNR, source variance as well
    # as sensor noise level (through cross-validation)
    for p in preproc_files:
        subject, condition = p.name.removesuffix(".set").split("_")
        logger.info(f"{subject=} | {condition=}")

        raw = load_raw(get_lemon_filename(subject, condition, params.lemon_bids))
        cond_save_path = subject_save_path / condition
        cond_save_path.mkdir(parents=True, exist_ok=True)

        if estimate_sensor_space_snr:
            logger.info("Inferring the sensor-space SNR")
            snr, spec_avg, freqs = infer_sensor_space_snr(
                raw, fmin=params.fmin, fmax=params.fmax
            )
            np.savez(
                cond_save_path / "sensor_space_snr.npz",
                snr=snr,
                spec_avg=spec_avg,
                freqs=freqs,
            )
            np.save(save_path / "freqs.npy", freqs)

        if estimate_source_power or estimate_noise_level_cv:
            fwd = load_head_model(
                "fsaverage", paths.subjects_dir, spacing=spacing, info=raw.info
            )
            inv = setup_inverse_operator(fwd, raw.info)

        if estimate_source_power:
            logger.info("Inferring source-space alpha power with eLORETA and LCMV")
            source_power_eloreta = infer_source_power(
                raw, fmin=params.fmin, fmax=params.fmax, method="eLORETA", inv=inv
            )
            source_power_lcmv = infer_source_power(
                raw, fmin=params.fmin, fmax=params.fmax, method="LCMV", fwd=fwd
            )
            np.savez(
                cond_save_path / f"source_space_power_{spacing}.npz",
                eloreta=source_power_eloreta,
                lcmv=source_power_lcmv,
            )

        if estimate_noise_level_cv:
            logger.info("Inferring the level of sensor noise via cross-validation")
            tmax = args.tmax or params.infer_tmax
            logger.info(f"Using the first {tmax} seconds of data")
            noise_level, best_lambda = infer_noise_level_cv(
                raw,
                fwd,
                tmax=tmax,
                fmin=params.fmin,
                fmax=params.fmax,
                lambdas=np.logspace(
                    params.infer_reg_min,
                    params.infer_reg_max,
                    num=params.infer_reg_steps,
                ),
                n_cv_splits=params.infer_n_cv_splits,
                random_state=params.seed,
            )

            np.savez(
                cond_save_path / f"cv_noise_{spacing}.npz",
                tmax=tmax,
                noise_level=noise_level,
                best_lambda=best_lambda,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.subject, args.spacing, args)
