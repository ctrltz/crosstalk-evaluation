import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.cm import viridis
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from ctfeval.config import paths
from ctfeval.io import init_subject
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.viz import (
    add_label,
    draw_text,
    set_plot_style,
    plot_data_on_brain,
    make_cropped_screenshot,
)
from ctfeval.viz_utils import squeeze_colormap

set_plot_style()


def main():
    # Load the head model
    fwd, *_ = init_subject(subject="fsaverage")
    src = fwd["src"]

    parc_folder = paths.theory / "parcellations"
    viridis_squeezed = squeeze_colormap(viridis)

    # Supplementary: plot maximal CTF ratios for multiple parcellations
    fig_supp, axes = plt.subplots(
        ncols=3,
        nrows=3,
        figsize=(10, 6),
        layout="constrained",
        gridspec_kw=dict(height_ratios=[1, 1, 0.05]),
    )
    axes[2, 0].axis("off")
    axes[2, 2].axis("off")
    for ax, p in zip(axes[:2, :].flatten(), PARCELLATIONS.values()):
        p.load("fsaverage", paths.subjects_dir)
        with np.load(parc_folder / f"{p.code}.npz") as data:
            max_ratios = data["max_ratios"]

        brain = plot_data_on_brain(
            max_ratios,
            src,
            kind="label",
            labels=p.labels,
            surface="inflated",
            cortex="low_contrast",
            hemi="split",
            size=(1600, 1600),
            views=["lat", "med"],
            view_layout="vertical",
            colorbar=False,
            colormap=viridis_squeezed,
            clim=dict(kind="value", lims=[0.0, 0.00001, 1.0]),
            transparent=True,
            ax_cbar=axes[2, 1],
            cbar_kws=dict(
                orientation="horizontal", label="Upper limit of the CTF ratio"
            ),
            cbar_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            cbar_outline=True,
        )
        brain.add_annotation(p.fs_name, color="black", borders=True)

        ax.set_title(p.name)
        make_cropped_screenshot(brain, ax=ax)

    fig_supp.savefig(
        paths.figures / "fig4supp_max_ratios.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_supp)

    # Set up the main figure
    fig = plt.figure(figsize=(10, 7), layout="constrained")
    g = GridSpec(
        nrows=4,
        ncols=4,
        figure=fig,
        width_ratios=[1, 1, 0.4, 0.6],
        height_ratios=[1, 0.07, 0.05, 1.1],
    )

    # Upper limits of CTF metrics
    sg = g[0, 0].subgridspec(nrows=2, ncols=2, height_ratios=[0.1, 0.9])

    ax_title = fig.add_subplot(sg[0, :])
    draw_text(ax_title, "Upper limits", fontsize="x-large")
    add_label(ax_title, "A")
    ax_title.axis("off")

    ax_max_ratio_cbar = fig.add_subplot(g[1, 0])
    for i, parcellation in enumerate(["DK", "Schaefer400"]):
        p = PARCELLATIONS[parcellation].load("fsaverage", paths.subjects_dir)
        with np.load(parc_folder / f"{parcellation}.npz") as data:
            max_ratios = data["max_ratios"]

        brain = plot_data_on_brain(
            max_ratios,
            src,
            kind="label",
            labels=p.labels,
            surface="inflated",
            cortex="low_contrast",
            hemi="lh",
            size=(800, 1600),
            views=["lat", "med"],
            view_layout="vertical",
            colorbar=False,
            colormap=viridis_squeezed,
            clim=dict(kind="value", lims=[0.0, 0.00001, 0.85]),
            transparent=True,
            ax_cbar=ax_max_ratio_cbar,
            cbar_kws=dict(
                label="CTF ratio",
                orientation="horizontal",
            ),
            cbar_outline=True,
            cbar_ticks=[0, 0.2, 0.4, 0.6, 0.8],
        )
        brain.add_annotation(p.fs_name, color="black", borders=True)

        ax = fig.add_subplot(sg[1, i])
        ax.set_title(p.name)
        make_cropped_screenshot(brain, ax=ax)

    # Achieved CTF ratios
    sg = g[0, 1:3].subgridspec(nrows=2, ncols=3, height_ratios=[0.1, 0.9])

    ax_title = fig.add_subplot(sg[0, :])
    draw_text(ax_title, "Achieved values", fontsize="x-large")
    add_label(ax_title, "B")
    ax_title.axis("off")

    ratios_list = []
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    data = np.load(parc_folder / "DK.npz")
    roi_methods = ["mean", "mean_flip", "centroid"]
    for i_method, roi_method in enumerate(roi_methods):
        ratios = data[f"ratios_eLORETA_{roi_method}"]
        brain = plot_data_on_brain(
            ratios,
            src,
            kind="label",
            labels=p.labels,
            surface="inflated",
            cortex="low_contrast",
            hemi="lh",
            size=(800, 1600),
            views=["lat", "med"],
            view_layout="vertical",
            colorbar=False,
            colormap=viridis_squeezed,
            clim=dict(kind="value", lims=[0.0, 0.00001, 0.85]),
            transparent=True,
        )
        brain.add_annotation(p.fs_name, color="black", borders=True)

        ax = fig.add_subplot(sg[1, i_method])
        ax.set_title(roi_method.replace("_", "-"))
        make_cropped_screenshot(brain, ax=ax)
        ratios_list.append(ratios)

    # Method with highest CTF ratio for each ROI
    # NOTE: add some offset (0.5) to reserve the value of 0 for 'Unknown' area
    ratios_all = np.vstack(ratios_list)
    best_method = np.argmax(ratios_all.T, axis=1) + 0.5

    sg = g[0, 3].subgridspec(nrows=2, ncols=1, height_ratios=[0.1, 0.9])

    ax_title = fig.add_subplot(sg[0])
    draw_text(ax_title, "Best method", fontsize="x-large")
    add_label(ax_title, "C")
    ax_title.axis("off")

    method_colors = matplotlib.colormaps["Dark2"].colors[:3]
    method_colormap = squeeze_colormap(ListedColormap(method_colors))

    ax_method = fig.add_subplot(sg[1])
    brain = plot_data_on_brain(
        best_method,
        src,
        kind="label",
        labels=p.labels,
        surface="inflated",
        cortex="low_contrast",
        hemi="lh",
        size=(800, 1600),
        views=["lat", "med"],
        view_layout="vertical",
        colorbar=False,
        colormap=method_colormap,
        clim=dict(kind="value", lims=[0.0, 0.00001, 3.0]),
        transparent=True,
    )
    brain.add_annotation(p.fs_name, color="black", borders=True)
    make_cropped_screenshot(brain, ax=ax_method)

    # Method legend
    legend_elements = []
    for color, roi_method in zip(method_colors, roi_methods):
        glyph = Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=roi_method.replace("_", "-"),
            markerfacecolor=color,
            markersize=15,
        )
        legend_elements.append(glyph)

    ax_method_legend = fig.add_subplot(g[1, 3])
    ax_method_legend.axis("off")
    ax_method_legend.legend(
        handles=legend_elements,
        loc="center",
        frameon=False,
        ncols=1,
        handlelength=1,
        columnspacing=1,
        borderaxespad=0,
    )

    # Max. CTF ratio vs. mean depth
    max_ratios = data["max_ratios"]
    mean_depth = data["mean_depth"]
    tick_formatter = ScalarFormatter()
    tick_formatter.set_scientific(False)

    ax_depth = fig.add_subplot(g[3, 0])
    ax_depth.scatter(mean_depth * 1e2, max_ratios)
    ax_depth.set_xlabel("Mean distance to the outer skull surface (cm)")
    ax_depth.set_ylabel("Upper limit of CTF ratio")
    ax_depth.set_xscale("log")
    ax_depth.set_yscale("log")
    ax_depth.xaxis.set_major_formatter(tick_formatter)
    ax_depth.yaxis.set_major_formatter(tick_formatter)
    ax_depth.set_xticks([3, 4, 5, 6])
    ax_depth.set_yticks([0.01, 0.05, 0.1, 0.5, 1])
    add_label(ax_depth, "D")

    # CTF ratio vs. ROI area
    roi_areas_cm2 = data["areas_cm2"]
    ax_area = fig.add_subplot(g[3, 1])
    ax_area.scatter(roi_areas_cm2, max_ratios)
    ax_area.set_xlabel("ROI area (cm$^2$)")
    ax_area.set_ylabel("Upper limit of CTF ratio")
    ax_area.set_xscale("log")
    ax_area.set_yscale("log")
    ax_area.xaxis.set_major_formatter(tick_formatter)
    ax_area.yaxis.set_major_formatter(tick_formatter)
    ax_area.set_xticks([2, 4, 8, 16, 32])
    ax_area.set_yticks([0.01, 0.05, 0.1, 0.5, 1])
    add_label(ax_area, "E")

    # CTF ratio vs. channel setup
    df_ratio = pd.read_csv(parc_folder / "DK_num_channels.csv")
    ax_setup = fig.add_subplot(g[3, 2:])
    palette = sns.color_palette(["#cccccc"], p.n_labels)  # same color for all lines

    sns.pointplot(
        data=df_ratio,
        x="ch_count",
        y="max_ratio",
        ax=ax_setup,
        hue="label",
        palette=palette,
        legend=False,
        markerfacecolor="tab:blue",
        markeredgecolor="tab:blue",
        markersize=3,
        lw=1,
    )
    ax_setup.set_ylim([0, 1])
    ax_setup.set_xlabel("Number of EEG sensors")
    ax_setup.set_ylabel("Upper limit of CTF ratio")
    add_label(ax_setup, "F")

    # Save the result
    fig.savefig(paths.figures / "fig4_ctf_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
