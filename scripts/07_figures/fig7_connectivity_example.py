import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from mne.viz import plot_brain_colorbar

from ctfeval.config import paths
from ctfeval.io import dict_from_npz, init_subject
from ctfeval.utils import data2stc
from ctfeval.viz import add_label, set_plot_style, make_cropped_screenshot

set_plot_style()


def main():
    # Load the prerequisites
    fwd, _, _, p = init_subject(parcellation="DK")
    src = fwd["src"]
    with np.load(paths.examples / "connectivity.npz") as data:
        target = data["target"]
        interference = data["interference"]
        freqs = data["freqs"]
        ctf = dict_from_npz(data, "ctf", transform=lambda w: data2stc(w, src))
        imcoh_t = dict_from_npz(data, "imcoh_t")
        imcoh_i = dict_from_npz(data, "imcoh_i")
    target_labels = [p[name] for name in target]
    interference_labels = [p[name] for name in interference]

    target_color = "#e13763"
    interference_color = "#0081c8"
    combined_color = "#5D0178"

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    gs = fig.add_gridspec(
        nrows=7,
        ncols=5,
        width_ratios=[1, 0.1, 1.5, 0.1, 1],
        height_ratios=[0.25, 1, 0.15, 0.25, 1, 0.1, 0.2],
    )

    # Plot CTFs for each considered ROI
    labels_to_plot = target_labels + interference_labels
    colors = [target_color] * len(target_labels) + [interference_color] * len(
        interference_labels
    )
    row_indices = [1, 1, 4, 4]
    col_indices = [0, 2, 0, 2]
    views = [["dor", "med"], ["lat", "med"], ["dor", "med"], ["lat", "med"]]
    panel_labels = ["A", "", "C", ""]
    panel_titles = [
        "Target ROI 1",
        "Target ROI 2",
        "Interfering ROI 1",
        "Interfering ROI 2",
    ]
    plot_grid = list(
        zip(
            labels_to_plot,
            colors,
            row_indices,
            col_indices,
            views,
            panel_labels,
            panel_titles,
        )
    )
    clim = dict(kind="value", lims=[0, 0.25, 1])
    cmap = "Greys"

    for label, color, row, col, views, panel_label, panel_title in plot_grid:
        brain = ctf[label.name].plot(
            subject="fsaverage",
            subjects_dir=paths.subjects_dir,
            size=(1600, 800),
            hemi=label.hemi,
            views=views,
            view_layout="horizontal",
            background="w",
            cortex="low_contrast",
            colormap=cmap,
            colorbar=False,
            clim=clim,
            transparent=False,
        )
        brain.add_label(label, color=color, borders=3)

        ax_label = fig.add_subplot(gs[row - 1, col])
        ax_label.text(
            0.5,
            0.5,
            panel_title,
            color=color,
            fontsize="large",
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )
        ax_label.axis("off")
        if panel_label:
            add_label(ax_label, panel_label, y=1.8)

        ax_brain = fig.add_subplot(gs[row, col])
        make_cropped_screenshot(brain, ax=ax_brain)

    # Colorbar for all CTFs
    sgs = gs[-2:, :4].subgridspec(
        nrows=3, ncols=3, width_ratios=[0.5, 1, 0.5], height_ratios=[0.5, 1, 0.5]
    )
    ax_ctf_cbar = fig.add_subplot(sgs[1, 1])
    cbar = plot_brain_colorbar(
        ax=ax_ctf_cbar,
        clim=clim,
        colormap=cmap,
        label="Normalized CTF",
        orientation="horizontal",
        transparent=False,
    )
    cbar.outline.set_visible(True)
    ax_ctf_cbar.set_xticks([0, 0.5, 1])

    # Iterate over all simulation cases + ground truth
    grid = list(
        zip(
            ["none", "target", "gt", "interference", "both"],
            ["gray", target_color, "gray", interference_color, combined_color],
            ["none", "target", "ground truth", "interference", "both"],
        )
    )

    # Target connection - ground-truth and estimated ImCoh
    ax_target = fig.add_subplot(gs[:2, 4])
    for case, color, label in grid:
        linestyle = "dashed" if case == "gt" else "solid"
        ax_target.plot(freqs, imcoh_t[case], c=color, ls=linestyle, label=label)
    ax_target.set_xlim([1, 25])
    ax_target.set_xlim([0, None])
    ax_target.set_ylim([0, None])
    ax_target.set_xlabel("Frequency (Hz)")
    ax_target.set_ylabel("abs(ImCoh)")
    ax_target.set_title("Target connection")
    add_label(ax_target, "B", x=-0.2)

    # Interfering connection - ground-truth and estimated ImCoh
    ax_interference = fig.add_subplot(gs[3:5, 4])
    for case, color, label in grid:
        linestyle = "dashed" if case == "gt" else "solid"
        ax_interference.plot(freqs, imcoh_i[case], c=color, ls=linestyle, label=label)
    ax_interference.set_xlim([1, 25])
    ax_interference.set_xlim([0, None])
    ax_interference.set_ylim([0, None])
    ax_interference.set_xlabel("Frequency (Hz)")
    ax_interference.set_ylabel("abs(ImCoh)")
    ax_interference.set_title("Interfering connection")
    add_label(ax_interference, "D", x=-0.2)

    # Legend
    legend_elements = []
    for case, color, label in grid:
        linestyle = "dashed" if case == "gt" else "solid"
        glyph = Line2D(
            [0, 1],
            [0, 0],
            color=color,
            ls=linestyle,
            label=label,
        )
        legend_elements.append(glyph)

    ax_legend = fig.add_subplot(gs[-2:, 4:])
    ax_legend.axis("off")
    ax_legend.legend(
        handles=legend_elements,
        loc="center",
        alignment="center",
        frameon=False,
        ncols=2,
        handlelength=1,
        columnspacing=1,
        borderaxespad=0,
    )

    # Save the result
    fig.savefig(
        paths.figures / "fig7_connectivity_example.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
