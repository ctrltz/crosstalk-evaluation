import matplotlib.pyplot as plt
import numpy as np
import string

from matplotlib.lines import Line2D
from mne.viz import plot_brain_colorbar

from roiextract.filter import SpatialFilter

from ctfeval.config import paths
from ctfeval.io import dict_from_npz, init_subject
from ctfeval.utils import data2stc
from ctfeval.viz import add_label, draw_text, set_plot_style, make_cropped_screenshot
from ctfeval.viz_utils import zoom_in

set_plot_style()

CORTEX = dict(colormap="Greys", vmin=-2, vmax=5)


def main():
    # Load the simulation results and other prerequisites
    fwd, _, lemon_info, p = init_subject(parcellation="DK")
    src = fwd["src"]
    with np.load(paths.examples / "activity.npz") as data:
        roi_methods = data["roi_methods"]
        targets = data["targets"]
        target_label = p[data["target_roi"]]
        sf_dict = dict_from_npz(data, "sf_dict", transform=lambda w: SpatialFilter(w))
        ctf_equal = dict_from_npz(data, "ctf", transform=lambda w: data2stc(w, src))
        corr_theory_equal = dict_from_npz(data, "corr_theory_equal")
        corr_theory_unequal = dict_from_npz(data, "corr_theory_unequal")
        corr_sim_equal = dict_from_npz(data, "corr_sim_equal")
        corr_sim_unequal = dict_from_npz(data, "corr_sim_unequal")

    # Parameters
    colors = [
        "#984ea3",  # outside of the ROI
        "#333333",  # within the ROI, closer to hand/foot area
        "#4daf4a",  # within the ROI, closer to face area
    ]
    descriptions = ["outside", "within, dorsal", "within, lateral"]
    threshold = 0.1
    lim = max([ctf.data.max() for ctf in ctf_equal.values()])

    # Set up the figure
    fig = plt.figure(figsize=(10, 6), layout="constrained")
    gs = fig.add_gridspec(
        nrows=5,
        ncols=10,
        width_ratios=[0.3, 1.5, 0.1, 1.75, 0.3, 0.7, 0.7, 0.3, 0.7, 0.7],
        height_ratios=[0.2, 1, 1, 1, 0.1],
    )

    # Labels
    label_grid = zip(
        [gs[0, 1], gs[0, 3], gs[0, 5:7], gs[0, 8:]],
        [
            "Spatial filter",
            "Cross-talk function",
            "Equal variance",
            "Unequal variance",
        ],
    )
    for i, (sgs, label) in enumerate(label_grid):
        ax_label = fig.add_subplot(sgs)
        draw_text(ax_label, label, fontsize="x-large")
        add_label(ax_label, string.ascii_uppercase[i])

    # Filters & CTF
    ax_sf_cbar = None
    ax_ctf_cbar = None
    for i_method, roi_method in enumerate(roi_methods):
        ax_label = fig.add_subplot(gs[i_method + 1, 0])
        draw_text(
            ax_label,
            roi_method.replace("_", "-"),
            facecolor="#dddddd",
            rotation="vertical",
            fontsize="x-large",
        )

        ax_sf = fig.add_subplot(gs[i_method + 1, 1])
        im, _ = sf_dict[roi_method].plot(
            lemon_info, sphere="eeglab", axes=ax_sf, show=False
        )

        if ax_sf_cbar is None:
            ax_sf_cbar = fig.add_subplot(gs[-1, 1])
            fig.colorbar(
                im, cax=ax_sf_cbar, orientation="horizontal", label="Filter weights"
            )

        ax_ctf = fig.add_subplot(gs[i_method + 1, 3])
        clim = dict(kind="value", lims=[0, threshold * lim, lim])
        cmap = "hot"
        brain = ctf_equal[roi_method].plot(
            background="white",
            surface="inflated",
            hemi="lh",
            views="lat",
            cortex=CORTEX,
            subject="fsaverage",
            subjects_dir=paths.subjects_dir,
            colorbar=False,
            colormap=cmap,
            clim=clim,
            transparent=True,
        )
        brain.add_label(target_label, color="black", borders=True)
        for target, color in zip(targets, colors):
            brain.add_foci(target, coords_as_verts=True, color=color, scale_factor=0.75)
        zoom_in(brain, target_label.name)
        make_cropped_screenshot(brain, ax=ax_ctf)

        if ax_ctf_cbar is None:
            ax_ctf_cbar = fig.add_subplot(gs[-1, 3])
            cbar = plot_brain_colorbar(
                ax=ax_ctf_cbar,
                clim=clim,
                colormap=cmap,
                label="Normalized CTF",
                orientation="horizontal",
                transparent=True,
            )
            cbar.outline.set_visible(True)
            ax_ctf_cbar.set_xticks([0, 0.5, 1])

        # Theory
        grid = zip(
            [5, 8], [corr_theory_equal[roi_method], corr_theory_unequal[roi_method]]
        )
        for grid_idx, ctf_targets in grid:
            ax_theory = fig.add_subplot(gs[i_method + 1, grid_idx])
            ax_theory.bar(
                [str(t) for t in targets],
                ctf_targets,
                edgecolor=colors,
                facecolor="white",
                lw=2,
            )
            ax_theory.set_ylabel("Explained variance")
            ax_theory.set_ylim([0, 1])
            ax_theory.tick_params(
                axis="x", which="both", labelbottom=False, bottom=False
            )
            if i_method == 0:
                ax_theory.set_title("Theory")
            if i_method == 2:
                ax_theory.set_xlabel("Vertices")

        # Simulation
        grid = zip([6, 9], [corr_sim_equal[roi_method], corr_sim_unequal[roi_method]])
        for grid_idx, coh_targets in grid:
            ax_sim = fig.add_subplot(gs[i_method + 1, grid_idx])
            ax_sim.bar([str(t) for t in targets], coh_targets**2, color=colors)
            ax_sim.set_ylabel("")
            ax_sim.set_yticklabels([])
            ax_sim.set_ylim([0, 1])
            ax_sim.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
            if i_method == 0:
                ax_sim.set_title("Simulation")
            if i_method == 2:
                ax_sim.set_xlabel("Vertices")

    fig.savefig(
        paths.figure_components / "fig3_minimal_example.svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # Vertex legend
    legend_elements = []
    for color, desc in zip(colors, descriptions):
        glyph = Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=desc,
            markerfacecolor=color,
            markersize=15,
        )
        legend_elements.append(glyph)

    fig_legend, ax_legend = plt.subplots()
    ax_legend.axis("off")
    ax_legend.legend(
        handles=legend_elements,
        loc="lower center",
        frameon=False,
        ncols=3,
        handlelength=1,
        columnspacing=1,
        borderaxespad=0,
    )
    fig_legend.savefig(
        paths.figure_components / "fig3_vertex_legend.svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig_legend)


if __name__ == "__main__":
    main()
