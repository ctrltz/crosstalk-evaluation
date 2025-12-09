import matplotlib.pyplot as plt
import numpy as np

from ctfeval.config import paths, params
from ctfeval.io import init_subject
from ctfeval.log import logger
from ctfeval.viz import add_label, draw_text, plot_data_on_brain, set_plot_style
from ctfeval.viz_utils import fill_band_area

set_plot_style()


def main():
    infer_path = paths.derivatives / "real_data" / "infer_params"
    infer_ga_path = infer_path / "GA"

    logger.info("Figure C - Inferred values of simulation parameters")
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.05)

    # Load results for sensor-space SNR
    with np.load(infer_ga_path / "sensor_space_snr.npz") as data:
        freqs = data["freqs"]
        snr_ec = data["snr_ec"]
        snr_eo = data["snr_eo"]
        spec_eo = data["spec_eo"]

    spec_eo_avg = np.nanmean(spec_eo, axis=0) * 1e12  # Convert to uV

    # Illustrate the approach
    logger.info("Panel A - Illustrating the approach for global SNR")
    ax_snr_approach = fig.add_subplot(gs[0, 0])
    band_color = "#dd0000"
    side_color = "#aaaaaa"

    disp_freqs = np.logical_and(freqs >= 1, freqs <= 45)
    ax_snr_approach.plot(freqs[disp_freqs], spec_eo_avg[disp_freqs], color="black")
    fill_band_area(
        ax_snr_approach, freqs, spec_eo_avg, params.fmin, params.fmax, band_color
    )
    fill_band_area(
        ax_snr_approach,
        freqs,
        spec_eo_avg,
        params.fmin - 3,
        params.fmin - 1,
        side_color,
    )
    fill_band_area(
        ax_snr_approach,
        freqs,
        spec_eo_avg,
        params.fmax + 1,
        params.fmax + 3,
        side_color,
    )
    ax_snr_approach.set_xlim([0, None])
    ax_snr_approach.set_ylim([0, None])
    ax_snr_approach.set_xlabel("Frequency (Hz)")
    ax_snr_approach.set_ylabel("PSD (uV^2/Hz)")
    add_label(ax_snr_approach, "A")

    # Show the histogram of SNR values for EC and EO conditions
    logger.info("Panel B - Histograms of SNR values")
    ax_snr_hist = fig.add_subplot(gs[0, 1])
    snr_values = 10 * np.log10([0.33, 1, 3])

    bins = np.linspace(-5, 15, num=20)
    ax_snr_hist.hist(10 * np.log10(snr_eo), bins=bins, label="EO", alpha=0.5)
    ax_snr_hist.hist(10 * np.log10(snr_ec), bins=bins, label="EC", alpha=0.5)
    for v in snr_values:
        ax_snr_hist.axvline(v, color="gray", ls="--")
    ax_snr_hist.legend(loc="upper right", frameon=False)
    ax_snr_hist.set_xlabel("SNR (dB)")
    ax_snr_hist.set_ylabel("Count")
    add_label(ax_snr_hist, "B")

    # Source power
    logger.info("Panel C - Spatial distribution of source power")
    fwd, *_ = init_subject()
    src = fwd["src"]

    var_ec = np.load(paths.precomputed / "var_oct6_EC_like.npy")
    var_ec /= np.max(var_ec)
    var_eo = np.load(paths.precomputed / "var_oct6_EO_like.npy")
    var_eo /= np.max(var_eo)

    sgs = gs[0, 2].subgridspec(ncols=2, nrows=5, height_ratios=[0.5, 2, 0.5, 2, 0.5])

    # EO condition
    ax_eo_label = fig.add_subplot(sgs[0, :])
    draw_text(ax_eo_label, "Eyes open", fontsize="large")
    add_label(ax_eo_label, "C")
    ax_eo_lat = fig.add_subplot(sgs[1, 0])
    plot_data_on_brain(
        var_eo,
        src,
        surface="pial_semi_inflated",
        hemi="rh",
        views="lat",
        make_screenshot=True,
        colorbar=False,
        ax=ax_eo_lat,
    )
    ax_eo_med = fig.add_subplot(sgs[1, 1])
    plot_data_on_brain(
        var_eo,
        src,
        surface="pial_semi_inflated",
        hemi="rh",
        views="med",
        make_screenshot=True,
        colorbar=False,
        ax=ax_eo_med,
    )

    # EC condition
    ax_ec_label = fig.add_subplot(sgs[2, :])
    draw_text(ax_ec_label, "Eyes closed", fontsize="large")
    ax_ec_lat = fig.add_subplot(sgs[3, 0])
    plot_data_on_brain(
        var_ec,
        src,
        surface="pial_semi_inflated",
        hemi="rh",
        views="lat",
        make_screenshot=True,
        colorbar=False,
        ax=ax_ec_lat,
    )
    ax_ec_med = fig.add_subplot(sgs[3, 1])
    plot_data_on_brain(
        var_ec,
        src,
        surface="pial_semi_inflated",
        hemi="rh",
        views="med",
        make_screenshot=True,
        colorbar=False,
        ax=ax_ec_med,
    )

    # Amplifier noise
    with np.load(infer_ga_path / "amplifier_noise.npz") as data:
        noise_level = data["noise_level"]
        spec_raw = data["spec_raw"]
        freqs_raw = data["freqs_raw"]
    bins = np.linspace(0, 1, num=21)
    psd_avg = np.nanmean(spec_raw, axis=0) * 1e12  # Convert to uV

    # Value of PSD at the 800 Hz was used as a proxy of amplifier noise
    famp_idx = np.argmin(np.abs(freqs_raw - params.infer_famp))
    famp_val = 10 * np.log10(psd_avg[famp_idx])

    # Illustrate the approach
    logger.info("Panel D - Illustrating the approach for amplifier noise level")
    ax_amp_approach = fig.add_subplot(gs[1, 0])

    ax_amp_approach.axhline(famp_val, c="gray", ls="-")
    ax_amp_approach.axvline(params.infer_famp, c="gray", ls=":")
    ax_amp_approach.plot(freqs_raw, 10 * np.log10(psd_avg))
    ax_amp_approach.set_xlim([1, 1250])
    ax_amp_approach.set_xscale("log")
    ax_amp_approach.set_xticks([1, 10, 100, 1000])
    ax_amp_approach.set_xticklabels([1, 10, 100, 1000])
    ax_amp_approach.set_xlabel("Frequency (Hz)")
    ax_amp_approach.set_ylabel("10 $\\cdot$ log$_{10}$(PSD)")
    add_label(ax_amp_approach, "D")

    # Histogram of amplifier noise level
    logger.info("Panel E - Histogram of amplifier noise level values")
    ax_amp_hist = fig.add_subplot(gs[1, 1])

    ax_amp_hist.hist(noise_level * 100, bins=bins)
    ax_amp_hist.set_xlim([0, None])
    ax_amp_hist.set_xlabel("Amplifier noise power / alpha power (%)")
    ax_amp_hist.set_ylabel("Count")
    add_label(ax_amp_hist, "E")

    # Level of sensor noise estimated via CV
    with np.load(infer_ga_path / "cv_noise_oct6.npz") as data:
        noise_level_ec = data["noise_level_ec"]
        noise_level_eo = data["noise_level_eo"]
    bins = np.linspace(0, 50, num=26)

    logger.info("Panel F - Histogram of CV-based noise level values")
    ax_cv_noise = fig.add_subplot(gs[1, 2])
    ax_cv_noise.hist(noise_level_eo * 100, bins=bins, alpha=0.5, label="EO")
    ax_cv_noise.hist(noise_level_ec * 100, bins=bins, alpha=0.5, label="EC")
    ax_cv_noise.legend(loc="upper right", frameon=False)
    ax_cv_noise.set_xlabel("Residual power / alpha power (%)")
    ax_cv_noise.set_ylabel("Count")
    add_label(ax_cv_noise, "F")

    # Save the result
    fig.savefig(
        paths.figures / "figCsupp_inferred_params.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
