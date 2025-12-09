import argparse
import numpy as np

from tqdm import tqdm

from ctfeval.config import paths, params
from ctfeval.datasets import get_lemon_subject_ids
from ctfeval.io import init_subject
from ctfeval.log import logger


parser = argparse.ArgumentParser()
parser.add_argument("--spacing", choices=["oct6", "ico4"], default="oct6")
parser.add_argument("--normalize-source-power", action="store_true")


def main(spacing, normalize_source_power):
    infer_path = paths.derivatives / "real_data" / "infer_params"
    infer_ga_path = infer_path / "GA"
    infer_ga_path.mkdir(exist_ok=True)

    # Load data: subject IDs, number of frequencies and sources
    subject_ids = get_lemon_subject_ids(params.lemon_bids)
    n_subjects = len(subject_ids)
    logger.info(f"Combining results of {n_subjects} subjects")

    conditions = ["EC", "EO"]
    freqs = np.load(infer_path / "freqs.npy")
    freqs_raw = np.load(infer_path / "freqs_raw.npy")
    n_freqs = freqs.size
    n_freqs_raw = freqs_raw.size

    fwd, *_ = init_subject(spacing=spacing)
    n_sources = fwd["nsource"]

    # Sensor-space SNR
    logger.info("Combining the obtained values of sensor-space SNR")
    snr_ec = np.full((n_subjects,), np.nan)
    spec_ec = np.full((n_subjects, n_freqs), np.nan)
    snr_eo = np.full((n_subjects,), np.nan)
    spec_eo = np.full((n_subjects, n_freqs), np.nan)

    for i, subject in enumerate(tqdm(subject_ids)):
        for condition, acc_snr, acc_spec in zip(
            conditions, [snr_ec, snr_eo], [spec_ec, spec_eo]
        ):
            cond_path = infer_path / subject / condition / "sensor_space_snr.npz"
            if not cond_path.exists():
                logger.warning(f"Could not find SNR results for {subject}")
                continue

            try:
                with np.load(cond_path) as data:
                    acc_snr[i] = data["snr"]
                    acc_spec[i, :] = data["spec_avg"]
                    freqs = data["freqs"]
            except Exception:
                logger.warning(f"Could not extract SNR results for {subject}")

    snr_ec_failed = np.isnan(snr_ec).sum()
    snr_eo_failed = np.isnan(snr_eo).sum()
    np.savez(
        infer_ga_path / "sensor_space_snr.npz",
        freqs=freqs,
        snr_ec=snr_ec,
        snr_eo=snr_eo,
        spec_ec=spec_ec,
        spec_eo=spec_eo,
    )

    # Source power
    logger.info("Combining the obtained values of source power")
    source_power_ec = np.full((n_subjects, n_sources), np.nan)
    source_power_eo = np.full((n_subjects, n_sources), np.nan)
    for i, subject in enumerate(tqdm(subject_ids)):
        for condition, acc in zip(conditions, [source_power_ec, source_power_eo]):
            cond_path = (
                infer_path / subject / condition / f"source_space_power_{spacing}.npz"
            )
            if not cond_path.exists():
                logger.warning(f"Could not find source power results for {subject}")
                continue

            try:
                with np.load(cond_path) as data:
                    source_power = np.squeeze(data["eloreta"])
                    if normalize_source_power:
                        # Make mean power equal to 1, thereby removing between-subject
                        # difference in total power
                        source_power /= np.mean(source_power)
                    acc[i, :] = source_power
            except Exception:
                logger.warning(f"Could not extract source power results for {subject}")

    source_power_ec_failed = np.any(np.isnan(source_power_ec), axis=1).sum()
    source_power_eo_failed = np.any(np.isnan(source_power_eo), axis=1).sum()
    source_power_ec_ga = np.nanmean(source_power_ec, axis=0)
    source_power_eo_ga = np.nanmean(source_power_eo, axis=0)

    np.savez(
        infer_ga_path / f"source_power_{spacing}.npz",
        source_power_ec=source_power_ec,
        source_power_eo=source_power_eo,
    )

    # Use resulting source power maps for simulations
    np.save(paths.precomputed / f"var_{spacing}_EC_like.npy", source_power_ec_ga)
    np.save(paths.precomputed / f"var_{spacing}_EO_like.npy", source_power_eo_ga)

    # Sensor-space noise: amplifier
    logger.info("Combining the obtained values of amplifier noise")
    amp_noise = np.full((n_subjects,), np.nan)
    noise_level = np.full((n_subjects,), np.nan)
    spec_raw = np.full((n_subjects, n_freqs_raw), np.nan)

    for i, subject in enumerate(tqdm(subject_ids)):
        cond_path = infer_path / subject / "amplifier_noise.npz"
        if not cond_path.exists():
            logger.warning(f"Could not find amplifier noise results for {subject}")
            continue

        try:
            with np.load(cond_path) as data:
                amp_noise[i] = data["amp_noise"]
                noise_level[i] = data["noise_level"]
                spec_raw[i, :] = np.squeeze(data["spec_avg"])
        except Exception:
            logger.warning(f"Could not extract amplifier noise results for {subject}")

    amp_noise_failed = np.isnan(noise_level).sum()
    np.savez(
        infer_ga_path / "amplifier_noise.npz",
        amp_noise=amp_noise,
        noise_level=noise_level,
        spec_raw=spec_raw,
        freqs_raw=freqs_raw,
    )

    # Sensor space noise: CV
    logger.info("Combining the obtained values of CV noise")
    noise_level_ec = np.full((n_subjects,), np.nan)
    noise_level_eo = np.full((n_subjects,), np.nan)
    best_lambda_ec = np.full((n_subjects,), np.nan)
    best_lambda_eo = np.full((n_subjects,), np.nan)

    grid = list(
        zip(
            conditions,
            [noise_level_ec, noise_level_eo],
            [best_lambda_ec, best_lambda_eo],
        )
    )

    for i, subject in enumerate(tqdm(subject_ids)):
        for condition, acc_noise, acc_lambda in grid:
            cond_path = infer_path / subject / condition / f"cv_noise_{spacing}.npz"
            if not cond_path.exists():
                logger.warning(f"Could not find CV noise results for {subject}")
                continue

            try:
                with np.load(cond_path) as data:
                    acc_noise[i] = data["noise_level"]
                    acc_lambda[i] = data["best_lambda"]
            except Exception:
                logger.warning(f"Could not extract CV noise results for {subject}")

    cv_noise_ec_failed = np.isnan(noise_level_ec).sum()
    cv_noise_eo_failed = np.isnan(noise_level_eo).sum()
    np.savez(
        infer_ga_path / f"cv_noise_{spacing}.npz",
        noise_level_ec=noise_level_ec,
        best_lambda_ec=best_lambda_ec,
        noise_level_eo=noise_level_eo,
        best_lambda_eo=best_lambda_eo,
    )

    # Final report
    logger.info("Overview of the GA analysis")
    logger.info(f"Sensor-space SNR, EC: failed for {snr_ec_failed} subjects")
    logger.info(f"Sensor-space SNR, EO: failed for {snr_eo_failed} subjects")
    logger.info(f"Source power, EC: failed for {source_power_ec_failed} subjects")
    logger.info(f"Source power, EO: failed for {source_power_eo_failed} subjects")
    logger.info(f"Amplifier noise: failed for {amp_noise_failed} subjects")
    logger.info(f"CV noise, EC: failed for {cv_noise_ec_failed} subjects")
    logger.info(f"CV noise, EO: failed for {cv_noise_eo_failed} subjects")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.spacing, args.normalize_source_power)
