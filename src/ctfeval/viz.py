import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns


from ctfeval.config import paths
from ctfeval.log import logger
from ctfeval.utils import data2stc, labeldata2data
from ctfeval.viz_utils import crop_screenshot, prepare_colormap, sort_labels


DEFAULT_PLOT_KWARGS = dict(
    views=["lat", "med"],
    hemi="split",
    size=(800, 800),
    subject="fsaverage",
    subjects_dir=paths.subjects_dir,
    background="w",
    colorbar=False,
    time_viewer=False,
    show_traces=False,
    transparent=False,
)


def set_plot_style():
    plt.style.use(paths.assets / "plots.mplstyle")


def add_label(ax, label, x=-0.1, y=1.15):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontfamily="Arial",
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )


def draw_text(ax, text, x=0.5, y=0.5, facecolor=None, **text_kwargs):
    kwargs = {"horizontalalignment": "center", "verticalalignment": "center"}
    kwargs.update(text_kwargs)
    ax.text(x, y, text, **kwargs)

    # NOTE: if we need to keep the background color, we cannot just call .axis('off')
    # Instead, we need to remove all elements ourselves
    if facecolor is not None:
        ax.set_facecolor(facecolor)
        for side in ["top", "bottom", "left", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labelleft=False,
            bottom=False,
            left=False,
        )
    else:
        ax.axis("off")


def make_cropped_screenshot(brain, crop=True, ax=None, close=True):
    # Make the screenshot
    screenshot = brain.screenshot()
    if close:
        brain.close()

    if crop:
        screenshot = crop_screenshot(screenshot)

    if ax is not None:
        ax.imshow(screenshot)
        ax.axis("off")

    return screenshot


def plot_data_on_brain(
    data,
    src,
    kind="voxel",
    labels=None,
    threshold=0.5,
    make_screenshot=False,
    close=True,
    ax=None,
    ax_cbar=None,
    cbar_kws=dict(),
    cbar_outline=False,
    cbar_ticks=None,
    **kwargs,
):
    limits, cmap = prepare_colormap(data, threshold)

    plot_kwargs = DEFAULT_PLOT_KWARGS.copy()
    plot_kwargs["clim"] = dict(kind="value", lims=limits)
    plot_kwargs["colormap"] = cmap
    plot_kwargs.update(kwargs)
    logger.debug(f"Plot kwargs: {plot_kwargs}")
    if kind == "label":
        assert labels is not None, "labels are required to plot label-wise data"
        data = labeldata2data(data, labels, src)
    brain = data2stc(data, src).plot(**plot_kwargs)

    if ax_cbar is not None:
        cbar = mne.viz.plot_brain_colorbar(
            ax=ax_cbar,
            clim=plot_kwargs["clim"],
            colormap=plot_kwargs["colormap"],
            transparent=plot_kwargs["transparent"],
            **cbar_kws,
        )
        if cbar_outline:
            cbar.outline.set_visible(True)
        if cbar_ticks is not None:
            ax_cbar.set_xticks(cbar_ticks)

    if not make_screenshot:
        return brain

    assert ax is not None
    make_cropped_screenshot(brain, ax=ax, close=close)


def fsaverage_brain(**kwargs):
    brain_kwargs = dict(
        subject="fsaverage",
        subjects_dir=paths.subjects_dir,
        background="w",
        cortex="low_contrast",
        surf="inflated",
        hemi="split",
        views=["lat", "med"],
        size=800,
    )
    brain_kwargs.update(kwargs)

    Brain = mne.viz.get_brain_class()
    brain = Brain(**brain_kwargs)
    return brain


def plot_connectivity_matrix(
    data,
    ratios,
    p,
    fig,
    gs,
    offset=0.02,
    panel_label=None,
    main_kws=dict(),
    side_kws=dict(),
    cbar_kws=dict(),
):
    order = sort_labels(p.label_names, ratios)

    sgs = gs.subgridspec(
        nrows=3, ncols=2, width_ratios=[offset, 1], height_ratios=[offset, 1, 0.05]
    )

    # Label
    if panel_label is not None:
        label_ax = fig.add_subplot(sgs[0, 0])
        label_ax.axis("off")
        add_label(label_ax, panel_label, x=-3, y=4.5)

    # Main axis
    main_ax = fig.add_subplot(sgs[1, 1])
    cbar_ax = fig.add_subplot(sgs[-1, 1])
    sns.heatmap(
        data[np.ix_(order, order)],
        ax=main_ax,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws=cbar_kws,
        square=True,
        xticklabels=[],
        yticklabels=[],
        **main_kws,
    )

    # Left side axis
    left_ax = fig.add_subplot(sgs[1, 0])
    sns.heatmap(
        ratios[order, np.newaxis],
        ax=left_ax,
        cbar=False,
        xticklabels=[],
        yticklabels=[],
        **side_kws,
    )
    left_ax.text(
        -0.2,
        0.75,
        "left hemisphere",
        rotation="vertical",
        transform=left_ax.transAxes,
        horizontalalignment="right",
        verticalalignment="center",
    )
    left_ax.text(
        -0.2,
        0.25,
        "right hemisphere",
        rotation="vertical",
        transform=left_ax.transAxes,
        horizontalalignment="right",
        verticalalignment="center",
    )

    # Top side axis
    top_ax = fig.add_subplot(sgs[0, 1])
    sns.heatmap(
        ratios[np.newaxis, order],
        ax=top_ax,
        cbar=False,
        xticklabels=[],
        yticklabels=[],
        **side_kws,
    )
    top_ax.text(
        0.25,
        1.05,
        "left hemisphere",
        transform=top_ax.transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    top_ax.text(
        0.75,
        1.05,
        "right hemisphere",
        transform=top_ax.transAxes,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
