import matplotlib.pyplot as plt
import mne
import numpy as np

from roiextract.filter import SpatialFilter
from tomlkit import document, dump

from ctfeval.config import paths
from ctfeval.io import init_subject
from ctfeval.log import logger
from ctfeval.viz import set_plot_style, plot_data_on_brain, make_cropped_screenshot
from ctfeval.toml import add_array

set_plot_style()


def main():
    logger.info("Loading the fsaverage head model")

    parcellation = "DK"
    fwd, inv, lemon_info, p = init_subject(parcellation=parcellation)
    src = fwd["src"]

    # We use right postcentral gyrus to illustrate the main point and additionally show
    # insula as an example of ROI with low CTF ratio
    test_label = p["postcentral-rh"]
    low_ratio_label = p["insula-rh"]

    logger.info("Preparing the spatial filters and CTFs")

    inv_method = "eLORETA"
    reg = 0.05
    roi_method = "mean_flip"
    clim = dict(kind="value", lims=[0, 0.2, 1])

    # Export common settings
    doc = document()
    doc.add("inv_method", inv_method)
    doc.add("reg", reg)
    doc.add("roi_method", roi_method)
    doc.add("parcellation", parcellation)
    doc.add("test_label", test_label.name)
    doc.add("low_ratio_label", low_ratio_label.name)

    sf = SpatialFilter.from_inverse(
        fwd,
        inv,
        test_label,
        inv_method=inv_method,
        lambda2=reg,
        roi_method=roi_method,
        subject="fsaverage",
        subjects_dir=paths.subjects_dir,
    )
    ctf = sf.get_ctf_fwd(fwd, mode="amplitude", normalize="max")
    ctf2 = sf.get_ctf_fwd(fwd, mode="power", normalize="max")

    sf_low = SpatialFilter.from_inverse(
        fwd,
        inv,
        low_ratio_label,
        inv_method=inv_method,
        lambda2=reg,
        roi_method=roi_method,
        subject="fsaverage",
        subjects_dir=paths.subjects_dir,
    )
    ctf2_low = sf_low.get_ctf_fwd(fwd, mode="power", normalize="max")

    logger.info("Preparing the components of Figure 1")

    # Normalize the weights of the spatial filter to be in [-1, 1] and plot
    sf.w = sf.w / np.abs(sf.w).max()
    fig, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    im, _ = sf.plot(lemon_info, sphere="eeglab", axes=ax, show=False)
    fig.savefig(paths.figure_components / "fig1_sf.svg", bbox_inches="tight")

    # CTF shows contributions in terms of amplitude
    fig, ax = plt.subplots(figsize=(3.5, 3))
    plot_data_on_brain(
        ctf.data,
        src,
        hemi="rh",
        views="lat",
        colorbar=False,
        transparent=False,
        make_screenshot=True,
        ax=ax,
    )
    fig.savefig(
        paths.figure_components / "fig1_ctf_amplitude.png", dpi=300, bbox_inches="tight"
    )

    # Element-wise squared CTF shows power contributions
    fig, ax = plt.subplots(figsize=(3.5, 3))
    brain = plot_data_on_brain(
        ctf2.data,
        src,
        hemi="rh",
        views="lat",
        colorbar=False,
        clim=clim,
        transparent=False,
    )
    make_cropped_screenshot(brain, ax=ax, close=False)
    fig.savefig(
        paths.figure_components / "fig1_ctf_power.png", dpi=300, bbox_inches="tight"
    )

    # Add ROI border and export again as an example of high CTF ratio
    fig, ax = plt.subplots(figsize=(3.5, 3))
    brain.add_label(test_label, color="black", borders=True)
    make_cropped_screenshot(brain, ax=ax)
    fig.savefig(
        paths.figure_components / "fig1_ctf_high.png", dpi=300, bbox_inches="tight"
    )

    # Insula as an example of low CTF ratio
    fig, ax = plt.subplots(figsize=(3.5, 3))
    brain = plot_data_on_brain(
        ctf2_low.data,
        src,
        hemi="rh",
        views="lat",
        colorbar=False,
        clim=clim,
        transparent=False,
    )
    brain.add_label(low_ratio_label, color="black", borders=True)
    make_cropped_screenshot(brain, ax=ax)
    fig.savefig(
        paths.figure_components / "fig1_ctf_low.png", dpi=300, bbox_inches="tight"
    )

    # Colormap (red-blue)
    fig, ax = plt.subplots(figsize=(0.5, 3.5))
    fig.colorbar(im, cax=ax)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["-", "+"])
    fig.savefig(paths.figure_components / "fig1_cmap_red_blue.svg", bbox_inches="tight")

    # Colormap (reds)
    fig, ax = plt.subplots(figsize=(0.5, 3.5))
    mne.viz.plot_brain_colorbar(
        ax, clim, colormap="Reds", orientation="vertical", label="", transparent=False
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["min", "max"])
    fig.savefig(paths.figure_components / "fig1_cmap_reds.svg", bbox_inches="tight")

    # Export the underlying values for the figure
    add_array(doc, "sf", sf.w, digits=4)
    add_array(doc, "ctf_amplitude", ctf.data, digits=4)
    add_array(doc, "ctf_power", ctf2.data, digits=4)
    add_array(doc, "ctf_low", ctf2_low.data, digits=4)

    with open(paths.toml / "fig1.toml", "w") as f:
        dump(doc, f)


if __name__ == "__main__":
    main()
