import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ctfeval.config import paths
from ctfeval.io import load_source_space
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.rift.models import NoSpreadModel, DistanceSpreadModel, CTFSpreadModel
from ctfeval.rift.viz import plot_coh_onestim
from ctfeval.viz import (
    set_plot_style,
    add_label,
    draw_text,
)

set_plot_style()


def main():
    # Load the prerequisites
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    ctf_all = np.load(paths.rift / "ctf_array.npy")
    ctfs = ctf_all[0, 0, :, :]  # use CTFs of an exemplary subject and pipeline
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)

    # Compute onestim and twostim values for exemplary vertices (ROI centroids)
    lh_vertno = p["pericalcarine-lh"].center_of_mass(
        subject="fsaverage", restrict_vertices=src, subjects_dir=paths.subjects_dir
    )
    rh_vertno = p["pericalcarine-rh"].center_of_mass(
        subject="fsaverage", restrict_vertices=src, subjects_dir=paths.subjects_dir
    )

    models = ["No RFS", "Distance-based RFS", "CTF-based RFS"]
    no_spread_onestim = NoSpreadModel(p.labels, lh_vertno, rh_vertno).pred_onestim()
    distance_onestim = DistanceSpreadModel(
        p.labels, src, lh_vertno, rh_vertno
    ).pred_onestim()
    ctf_onestim = CTFSpreadModel(
        p.labels, src, lh_vertno, rh_vertno, gamma=100
    ).pred_onestim(ctfs)
    distance_onestim = 1.0 / distance_onestim  # for illustration purposes

    no_spread_twostim = NoSpreadModel(p.labels, lh_vertno, rh_vertno).pred_twostim()
    distance_twostim = DistanceSpreadModel(
        p.labels, src, lh_vertno, rh_vertno
    ).pred_twostim()
    ctf_twostim = CTFSpreadModel(
        p.labels, src, lh_vertno, rh_vertno, gamma=100
    ).pred_twostim(ctfs)
    distance_twostim = 1.0 / distance_twostim  # for illustration purposes

    # Plot and save the results
    fig = plt.figure(figsize=(10, 6.5), layout="constrained")
    gs = fig.add_gridspec(
        nrows=5,
        ncols=5,
        width_ratios=[1, 0.1, 1, 0.1, 1],
        height_ratios=[0.2, 0.9, 0.1, 0.2, 1.3],
    )

    ax_brain_stim_label = fig.add_subplot(gs[0, :])
    add_label(ax_brain_stim_label, "A", x=0)
    draw_text(
        ax_brain_stim_label,
        "Brain-stimulus coherence",
        fontsize="x-large",
        fontweight="bold",
    )
    for i, (values, model) in enumerate(
        zip([no_spread_onestim, distance_onestim, ctf_onestim], models)
    ):
        ax = fig.add_subplot(gs[1, i * 2])
        plot_coh_onestim(values, src, p, lh_vertno, rh_vertno, ax=ax)
        ax.set_title(model)

    ax_brain_brain_label = fig.add_subplot(gs[3, :])
    add_label(ax_brain_brain_label, "B", x=0)
    draw_text(
        ax_brain_brain_label, "Brain-brain ImCoh", fontsize="x-large", fontweight="bold"
    )
    for i, (values, model) in enumerate(
        zip([no_spread_twostim, distance_twostim, ctf_twostim], models)
    ):
        ax = fig.add_subplot(gs[4, i * 2])
        sns.heatmap(
            values, square=True, cbar=False, xticklabels=[], yticklabels=[], ax=ax
        )
        ax.set_title(model)

    fig.savefig(paths.figures / "figDsupp_rfs_models.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
