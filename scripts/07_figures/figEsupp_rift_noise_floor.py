import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.patches import Rectangle

from ctfeval.config import paths, params
from ctfeval.datasets import rift_subfolder
from ctfeval.io import init_subject
from ctfeval.log import logger
from ctfeval.viz import add_label, draw_text, plot_data_on_brain, set_plot_style

set_plot_style()


def noise_floor_onestim(coh_real, coh_noise, coh_threshold, methods, src, p, n_bins=21):
    logger.info("Plotting the noise floor for single-stimulus condition")

    # Histogram bins
    max_coh = np.ceil(coh_real.max() / 0.05) * 0.05  # ceil with 0.05 tolerance
    bins = np.linspace(0, max_coh, num=n_bins)

    # Create the figure
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(10, 8),
        layout="constrained",
        gridspec_kw=dict(width_ratios=[2, 1, 0.1]),
    )

    # Histogram of noise values
    ax_noise = axes[0, 0]
    ax_noise.hist(coh_noise, bins, density=True, color="#dddddd", fill=False)
    ax_noise.axvline(coh_threshold, c="red", ls=":")
    ax_noise.set_xlim([0, max_coh])
    ax_noise.set_xlabel("Coherence")
    ax_noise.set_ylabel("Density")

    ax_noise_desc = axes[0, 2]
    draw_text(
        ax_noise_desc,
        "noise",
        0.5,
        0.5,
        facecolor=None,
        fontsize="x-large",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=270,
    )
    rect = Rectangle(
        (0, 0),
        1,
        1,
        color="#dddddd",
        lw=3,
        fill=False,
        transform=ax_noise_desc.transAxes,
        clip_on=False,
    )

    ax_noise_desc.add_patch(rect)

    axes[0, 1].axis("off")

    for i_method, roi_method in enumerate(methods):
        ax_method = axes[i_method + 1, 2]
        draw_text(
            ax_method,
            roi_method.replace("_", "-"),
            0.5,
            0.5,
            facecolor="#dddddd",
            fontsize="x-large",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=270,
        )

        ax_hist = axes[i_method + 1, 0]
        ax_hist.hist(coh_real[i_method, :], bins, color="#dddddd", density=True)
        ax_hist.axvline(coh_threshold, c="red", ls=":")
        ax_hist.set_xlim([0, max_coh])
        ax_hist.set_ylim([0, None])
        ax_hist.set_xlabel("Coherence")
        ax_hist.set_ylabel("Density")
        ax_hist.set_yticks([0, 5, 10])

        sig = (coh_real[i_method, :] >= coh_threshold).astype(int)
        n_sig = np.sum(sig)

        ax_brain = axes[i_method + 1, 1]
        plot_data_on_brain(
            sig,
            src,
            kind="label",
            labels=p.labels,
            colormap="Reds",
            make_screenshot=True,
            ax=ax_brain,
            colorbar=False,
        )
        ax_brain.set_title(f"{n_sig} / {p.n_labels} significant ROIs")

    add_label(axes[0, 0], "A")
    add_label(axes[1, 0], "B")

    # Save the results
    output_path = paths.figures / "figEsupp_rift_noise_floor_onestim.png"
    logger.info(f"Saving the figure to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def noise_floor_twostim(
    imcoh_real, imcoh_noise, imcoh_threshold, methods, src, p, n_bins=16
):
    logger.info("Plotting the noise floor for two-stimuli condition")

    # Histogram bins
    max_imcoh = np.ceil(imcoh_real.max() / 0.05) * 0.05  # ceil with 0.05 tolerance
    bins = np.linspace(0, max_imcoh, num=n_bins)

    # Create the figure
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    gs = fig.add_gridspec(nrows=4, ncols=4, width_ratios=[2, 1, 1, 0.15])

    # Noise
    ax_noise = fig.add_subplot(gs[0, 0])
    ax_noise.hist(imcoh_noise, bins, density=True, color="#dddddd", fill=False)
    ax_noise.axvline(imcoh_threshold, c="red", ls=":")
    ax_noise.set_xlim([0, max_imcoh])
    ax_noise.set_xlabel("abs(ImCoh)")
    ax_noise.set_ylabel("Density")
    add_label(ax_noise, "A")

    ax_noise_desc = fig.add_subplot(gs[0, -1])
    draw_text(
        ax_noise_desc,
        "noise",
        x=0.5,
        y=0.5,
        facecolor=None,
        fontsize="x-large",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=270,
    )
    rect = Rectangle(
        (0, 0),
        1,
        1,
        color="#dddddd",
        lw=3,
        fill=False,
        transform=ax_noise_desc.transAxes,
        clip_on=False,
    )

    ax_noise_desc.add_patch(rect)

    for i_method, roi_method in enumerate(methods):
        ax_method = fig.add_subplot(gs[i_method + 1, -1])
        draw_text(
            ax_method,
            roi_method.replace("_", "-"),
            x=0.5,
            y=0.5,
            facecolor="#dddddd",
            fontsize="x-large",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=270,
        )

        absimcoh_method = imcoh_real[i_method, :, :].copy().squeeze()
        triu_indices = np.triu_indices_from(absimcoh_method, 1)

        ax_hist = fig.add_subplot(gs[i_method + 1, 0])
        ax_hist.hist(absimcoh_method[triu_indices], bins, color="#dddddd", density=True)
        ax_hist.axvline(imcoh_threshold, c="red", ls=":")
        ax_hist.set_xlim([0, max_imcoh])
        ax_hist.set_ylim([0, None])
        ax_hist.set_xlabel("abs(ImCoh)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_yticks([0, 10, 20])

        if i_method == 0:
            add_label(ax_hist, "B")

        ax_orig = fig.add_subplot(gs[i_method + 1, 1])
        sns.heatmap(
            absimcoh_method,
            xticklabels=[],
            yticklabels=[],
            square=True,
            ax=ax_orig,
            cbar=False,
        )

        sig = absimcoh_method >= imcoh_threshold
        n_sig = np.sum(sig[triu_indices])

        ax_masked = fig.add_subplot(gs[i_method + 1, 2])
        absimcoh_masked = absimcoh_method.copy()
        absimcoh_masked[~sig] = np.nan
        absimcoh_masked[sig] = 1.0
        sns.heatmap(
            absimcoh_masked,
            xticklabels=[],
            yticklabels=[],
            vmin=0.0,
            vmax=1.0,
            square=True,
            cmap="Reds",
            ax=ax_masked,
            cbar=False,
        )
        ax_masked.set_title(f"{n_sig} / {triu_indices[0].size} edges")

        # Show a border around the heatmap
        ax_masked.axhline(y=0, color="k", linewidth=1)
        ax_masked.axhline(y=absimcoh_masked.shape[1], color="k", linewidth=1)
        ax_masked.axvline(x=0, color="k", linewidth=1)
        ax_masked.axvline(x=absimcoh_masked.shape[0], color="k", linewidth=1)

    # Save the results
    output_path = paths.figures / "figEsupp_rift_noise_floor_twostim.png"
    logger.info(f"Saving the figure to {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def main():
    # Paths
    tag_folder_onestim = paths.rift / rift_subfolder(
        params.rift_onestim.tagging_type, params.rift_onestim.random_phases
    )
    tag_folder_twostim = paths.rift / rift_subfolder(
        params.rift_twostim.tagging_type, params.rift_twostim.random_phases
    )
    theory_path_onestim = tag_folder_onestim / "theory"
    theory_path_twostim = tag_folder_twostim / "theory"
    brain_stim_path = tag_folder_onestim / "brain_stimulus"
    brain_brain_path = tag_folder_twostim / "brain_brain"

    # Load common resources
    fwd, _, _, p = init_subject(parcellation="DK")
    src = fwd["src"]
    methods = ["mean", "mean_flip", "centroid"]

    logger.info("===== Brain-stimulus coherence =====")

    # Load noise floor
    with np.load(theory_path_onestim / "noise_floor_coh_onestim.npz") as data:
        coh_noise = data["conn_noise"]
        coh_threshold = data["conn_threshold"]

    # Brain-stimulus coherence, 1 stimulus
    abscoh_1stim = np.load(brain_stim_path / "onestim_avg_abscoh_stim.npy")
    noise_floor_onestim(abscoh_1stim, coh_noise, coh_threshold, methods, src, p)

    logger.info("===== Brain-brain ImCoh =====")

    # Load noise floor
    with np.load(theory_path_twostim / "noise_floor_imcoh_twostim.npz") as data:
        coh_noise = data["conn_noise"]
        coh_threshold = data["conn_threshold"]

    # Brain-brain ImCoh, 2 stimuli
    absimcoh_2stim = np.load(brain_brain_path / "twostim_avg_absimcoh.npy")
    noise_floor_twostim(absimcoh_2stim, coh_noise, coh_threshold, methods, src, p)


if __name__ == "__main__":
    main()
