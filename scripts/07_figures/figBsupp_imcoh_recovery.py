import matplotlib.pyplot as plt
import numpy as np

from ctfeval.config import paths
from ctfeval.io import init_subject
from ctfeval.viz import (
    add_label,
    set_plot_style,
    plot_connectivity_matrix,
    plot_data_on_brain,
)
from ctfeval.viz_utils import sort_labels

set_plot_style()


def main(roi_method="mean_flip"):
    # Load the results
    recovery_file = (
        paths.theory / "connectivity_estimation" / f"recovery_{roi_method}.npy"
    )
    recovery = np.load(recovery_file)
    with np.load(paths.theory / "parcellations" / "DK.npz") as data:
        ratios = data[f"ratios_eLORETA_{roi_method}"]

    # Create the layout
    fig = plt.figure(figsize=(10, 4), layout="constrained")
    gs = fig.add_gridspec(
        nrows=1, ncols=5, width_ratios=[1, 0.15, 1.6, 0.15, 1.5], wspace=0.1
    )

    # Load the head model
    fwd, _, _, p = init_subject(parcellation="DK")

    # Get recovery of own and other connections
    recovery_self = np.full((p.n_labels, p.n_labels), np.nan)
    recovery_ratio = np.full((p.n_labels, p.n_labels), np.nan)
    for i_label in range(p.n_labels):
        for j_label in range(p.n_labels):
            recovery_edge = np.squeeze(recovery[i_label, j_label, :, :])
            others_mask = np.full(recovery_edge.shape, True)
            others_mask[np.diag_indices_from(others_mask)] = False
            others_mask[i_label, j_label] = False
            others_mask[j_label, i_label] = False

            recovery_self[i_label, j_label] = recovery_edge[i_label, j_label]
            max_other = recovery_edge[others_mask].max()
            recovery_ratio[i_label, j_label] = (
                recovery_self[i_label, j_label] / max_other
            )

    # Plot the color scheme (average ratio across homologous pairs of ROIs)
    sort_values = sort_labels(p.label_names, ratios, return_values=True)

    src = fwd["src"]
    sgs = gs[0].subgridspec(nrows=2, ncols=1, height_ratios=[1, 0.05])
    ax = fig.add_subplot(sgs[0])
    ax_cbar = fig.add_subplot(sgs[1])
    plot_data_on_brain(
        sort_values,
        src,
        kind="label",
        labels=p.labels,
        make_screenshot=True,
        clim=dict(kind="value", lims=[0, 0.3, 0.6]),
        ax=ax,
        ax_cbar=ax_cbar,
        views=["lat", "med"],
        hemi="lh",
        view_layout="vertical",
        colormap="Grays",
        cbar_kws=dict(orientation="horizontal", label="CTF ratio"),
        cbar_ticks=[0, 0.2, 0.4, 0.6],
    )
    add_label(ax, "A")

    # Plot the recovery of own connections
    plot_connectivity_matrix(
        recovery_self,
        ratios,
        p,
        fig,
        gs[2],
        main_kws=dict(cmap="rocket"),
        side_kws=dict(cmap="Grays", vmin=0, vmax=0.6),
        cbar_kws=dict(orientation="horizontal", label="Contribution weights"),
        panel_label="B",
    )

    # Plot the own/other recovery ratio
    plot_connectivity_matrix(
        recovery_ratio,
        ratios,
        p,
        fig,
        gs[4],
        main_kws=dict(cmap="RdBu_r", center=1),
        side_kws=dict(cmap="Grays", vmin=0, vmax=0.6),
        cbar_kws=dict(orientation="horizontal", label="Ratio of contribution weights"),
        panel_label="C",
    )

    # Save the result
    fig.savefig(
        paths.results / "figures" / "figBsupp_imcoh_recovery.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
