import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ctfeval.config import paths, params
from ctfeval.log import logger
from ctfeval.viz import add_label, draw_text

plt.style.use(paths.assets / "plots.mplstyle")


def main(parcellation="DK", n_bins=5):
    # Create the figure
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    gs = fig.add_gridspec(
        nrows=5,
        ncols=4,
        width_ratios=[1, 1, 1, 0.05],
        height_ratios=[1, 0.1, 1, 0.1, 1],
    )

    # Load the grand-average cross-spectra
    data_path = paths.derivatives / "real_data" / "spurious_coherence"
    with np.load(data_path / "grand_average" / "results_mean.npz") as data:
        freqs = data["freqs"]
        mean_coh = data["coh"]
        mean_sc_perm_alpha = data["sc_perm_alpha"]

    roi_methods = ["mean", "mean_flip", "centroid"]
    pipelines = [f"eLORETA_{roi_method}" for roi_method in roi_methods]

    # Load the theoretical results
    sc_theory = {}

    for condition in params.sc_conditions:
        sc_theory[condition] = []

        with np.load(
            paths.derivatives / "theory" / "spurious_coherence" / f"sc_{condition}.npz"
        ) as data:
            for method in roi_methods:
                sc_theory[condition].append(data[method])

        sc_theory[condition] = np.stack(sc_theory[condition], axis=0)
    sc_theory = np.stack(list(sc_theory.values()), axis=0)

    # Plot EC results in the main text
    cond_idx = params.sc_conditions.index("EC")

    # Bin edges by distance
    distances = np.load(data_path / "comparison" / f"distances_{parcellation}.npy")
    edges = np.triu_indices_from(distances, 1)
    values = 100 * distances[edges]  # convert to cm

    q = np.linspace(0, 100, num=n_bins + 1)
    perc = np.percentile(values, q)

    # Panel A (inset): histogram of distances between ROIs
    sgs = gs[0, :2].subgridspec(nrows=1, ncols=4, width_ratios=[1.3, 0.1, 0.9, 0.45])
    ax_coh = fig.add_subplot(sgs[0])
    ax_dist_hist = inset_axes(
        ax_coh, width="40%", height="30%", loc="upper right", borderpad=0
    )
    binned_edges = []
    for i, (lo, hi) in enumerate(zip(perc[:-1], perc[1:])):
        # Select the distances that fall in the current bin
        bin_mask = np.logical_and(values >= lo, values < hi)

        # Select the corresponding connections
        rows, cols = edges
        bin_edges = (rows[bin_mask], cols[bin_mask])
        binned_edges.append(bin_edges)

        # Plot to check
        logger.info(f"Bin {i+1}: [{lo:.2f}, {hi:.2f}] cm, {sum(bin_mask)} edges")
        ax_dist_hist.hist(
            100 * distances[bin_edges],
            np.linspace(0, values.max(), num=30),
            orientation="vertical",
        )

    ax_dist_hist.set_xlim([0, None])
    ax_dist_hist.set_xticks([0, 6, 12])
    ax_dist_hist.set_yticks([0, 100])
    ax_dist_hist.set_xlabel("Distance (cm)")
    ax_dist_hist.set_ylabel("Count")

    # Panel A: coherence spectra split by distance
    pipeline_idx = pipelines.index("eLORETA_mean_flip")
    n_freqs = mean_coh.shape[2]

    for i_bin, bin_edges in enumerate(binned_edges):
        mean_coh_spectra = np.zeros((n_freqs,))
        for i_freq, _ in enumerate(freqs):
            mean_coh_spectra[i_freq] = np.mean(
                np.squeeze(mean_coh[cond_idx, pipeline_idx, i_freq, :, :])[bin_edges]
            )
        ax_coh.plot(freqs, mean_coh_spectra)

    ax_coh.set_xlim([1, 45])
    ax_coh.set_ylim([0, 0.8])
    ax_coh.set_xlabel("Frequency (Hz)")
    ax_coh.set_ylabel("Coherence")
    add_label(ax_coh, "A")

    # Panel B: model comparison
    # raw
    corr_df = pd.read_csv(data_path / "comparison" / "comparison_raw.csv")
    corr_df_mean = (
        corr_df.groupby(["model", "condition"])["corr"].agg("mean").reset_index()
    )

    ax_raw = fig.add_subplot(sgs[2])
    sns.barplot(
        corr_df_mean,
        x="model",
        y="corr",
        hue="condition",
        order=["Distance", "CTF"],
        ax=ax_raw,
    )
    ax_raw.set_xlabel("Model")
    ax_raw.xaxis.set_label_coords(0.75, -0.15)
    ax_raw.set_ylabel("Correlation")
    ax_raw.set_ylim([0, 1])
    ax_raw.set_title("raw")
    ax_raw.legend(frameon=False)
    add_label(ax_raw, "B", x=-0.25)

    # delta
    corr_delta_df = pd.read_csv(data_path / "comparison" / "comparison_delta.csv")
    corr_delta_df_mean = (
        corr_delta_df.groupby(["model", "condition"])["corr"].agg("mean").reset_index()
    )

    ax_delta = fig.add_subplot(sgs[3])
    sns.barplot(
        corr_delta_df_mean,
        x="model",
        y="corr",
        hue="condition",
        ax=ax_delta,
        legend=False,
    )
    ax_delta.set_xlabel("")
    ax_delta.set_ylabel("")
    ax_delta.set_yticklabels([])
    ax_delta.set_ylim([0, 1])
    ax_delta.set_title("delta")

    # Panel C: pairwise distances, best fit
    distances_fit = np.load(
        data_path / "comparison" / f"distances_{parcellation}_fit.npy"
    )
    distances_fit_cond = np.squeeze(distances_fit[cond_idx, :, :])
    ax_dist_mat = fig.add_subplot(gs[0, 2])
    sns.heatmap(
        distances_fit_cond,
        xticklabels=[],
        yticklabels=[],
        vmin=0,
        vmax=1,
        square=True,
        ax=ax_dist_mat,
        cbar=False,
    )
    ax_dist_mat.set_title("Distance (best fit)")
    add_label(ax_dist_mat, "C")

    # Panel D: theoretical estimates based on CTF
    ax_theory_label = fig.add_subplot(gs[1, :])
    draw_text(
        ax_theory_label,
        "Expected amount of spurious coherence (based on CTF)",
        y=0.5,
        fontsize="x-large",
        fontweight="bold",
    )
    add_label(ax_theory_label, "D", x=-0.025)

    cbar_ax = fig.add_subplot(gs[2, -1])
    for i_method, roi_method in enumerate(roi_methods):
        ax = fig.add_subplot(gs[2, i_method])
        sc_theory_method = np.squeeze(sc_theory[cond_idx, i_method, :, :])
        sns.heatmap(
            sc_theory_method,
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=[],
            yticklabels=[],
            ax=ax,
            cbar_ax=cbar_ax,
            cbar_kws=dict(label="Spurious coherence"),
        )
        ax.set_title("eLORETA + " + roi_method.replace("_", "-"))

    # Panel E: permutation results
    ax_perm_label = fig.add_subplot(gs[3, :])
    draw_text(
        ax_perm_label,
        "Permutation-based estimates of spurious coherence in real data",
        y=0.75,
        fontsize="x-large",
        fontweight="bold",
    )
    add_label(ax_perm_label, "E", x=-0.025)

    for i_method, roi_method in enumerate(roi_methods):
        ax = fig.add_subplot(gs[4, i_method])
        sc_real_method = np.squeeze(mean_sc_perm_alpha[cond_idx, i_method, :, :])
        sns.heatmap(
            sc_real_method,
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=[],
            yticklabels=[],
            ax=ax,
            cbar=False,
        )
        ax.set_title("eLORETA + " + roi_method.replace("_", "-"))

    # Save the result
    fig.savefig(
        paths.figures / "fig6_spurious_coherence.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
