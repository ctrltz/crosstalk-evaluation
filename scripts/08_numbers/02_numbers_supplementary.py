import numpy as np
import pandas as pd

from ctfeval.config import paths, params
from ctfeval.log import logger
from ctfeval.tex import number2tex, section2tex


def export_literature_review(f):
    logger.info("Literature review")
    df_included = pd.read_csv(f"{paths.review}/included.csv")

    f.write(section2tex("Literature review"))
    f.write(number2tex("reviewMinFreq", params.review_min_count, digits=0))
    f.write(number2tex("reviewNumIncluded", df_included.PMID.unique().size, digits=0))
    f.write(number2tex("reviewMinYear", df_included.Year.min(), digits=0))
    f.write(number2tex("reviewMaxYear", df_included.Year.max(), digits=0))


def export_inferred_params(f):
    logger.info("Inferred simulation parameters")
    infer_path = paths.derivatives / "real_data" / "infer_params"
    infer_ga_path = infer_path / "GA"

    # Config
    f.write("\n")
    f.write(section2tex("Inferred simulation parameters - config"))
    f.write(number2tex("inferFamp", params.infer_famp, digits=0))
    f.write(number2tex("inferCVfolds", params.infer_n_cv_splits, digits=0))
    f.write(number2tex("inferCVMinutes", params.infer_tmax // 60, digits=0))
    f.write(number2tex("inferCVregMin", 10**params.infer_reg_min, digits=4))
    f.write(number2tex("inferCVregMax", 10**params.infer_reg_max, digits=0))
    f.write(number2tex("inferCVregSteps", params.infer_reg_steps, digits=0))

    # Sensor space SNR
    with np.load(infer_ga_path / "sensor_space_snr.npz") as data:
        snr_ec = data["snr_ec"]
        snr_eo = data["snr_eo"]

    f.write("\n")
    f.write(section2tex("Inferred simulation parameters - global SNR"))
    for cond, snr_values in zip(["EC", "EO"], [snr_ec, snr_eo]):
        median_snr = np.nanmedian(snr_values)
        median_snr_dB = 10 * np.log10(median_snr)
        f.write(number2tex(f"infer{cond}SNR", median_snr, digits=1))
        f.write(number2tex(f"infer{cond}SNRdB", median_snr_dB, digits=1))

    # Sensor noise, CV
    with np.load(infer_ga_path / "cv_noise_oct6.npz") as data:
        noise_level_ec = data["noise_level_ec"]
        noise_level_eo = data["noise_level_eo"]

    f.write("\n")
    f.write(section2tex("Inferred simulation parameters - sensor noise via CV"))
    for cond, noise_level in zip(["EC", "EO"], [noise_level_ec, noise_level_eo]):
        median_noise_level = np.nanmedian(noise_level) * 100
        f.write(number2tex(f"infer{cond}noiseCV", median_noise_level, digits=1))


def main():
    f = open(paths.numbers / "numbers_supplementary.tex", "w+")
    try:
        export_literature_review(f)
        export_inferred_params(f)
    finally:
        f.close()


if __name__ == "__main__":
    main()
