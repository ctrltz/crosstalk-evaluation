import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ctfeval.rift.fit_evaluate import fit_and_predict
from ctfeval.rift.models import get_theory
from ctfeval.rift.utils import preproc_onestim, preproc_twostim, matrix_from_triu
from ctfeval.viz import (
    plot_data_on_brain,
    make_cropped_screenshot,
    add_label,
    draw_text,
)


def plot_coh_onestim(
    values,
    src,
    p,
    lh_vertno,
    rh_vertno,
    ax,
    colormap="viridis",
    vert_color="black",
    **kwargs,
):
    # Plot coherence as a brain map
    brain = plot_data_on_brain(
        values, src, kind="label", labels=p.labels, colormap=colormap, **kwargs
    )

    # Show ground-truth locations
    if lh_vertno is not None:
        brain.add_foci(lh_vertno, coords_as_verts=True, hemi="lh", color=vert_color)
    if rh_vertno is not None:
        brain.add_foci(rh_vertno, coords_as_verts=True, hemi="rh", color=vert_color)

    # Make a screenshot of the result
    make_cropped_screenshot(brain, ax=ax)


def plot_imcoh_twostim(values, p, vmin, center, vmax, ax):
    sns.heatmap(
        matrix_from_triu(values, p.n_labels),
        vmin=vmin,
        center=center,
        vmax=vmax,
        cbar=False,
        ax=ax,
        square=True,
        xticklabels=[],
        yticklabels=[],
    )


def plot_fit_extended(
    kind,
    ctfs,
    src,
    p,
    methods,
    coh_real,
    coh_threshold,
    metric_fun,
    model,
    lh_pick,
    rh_pick,
    gamma,
    vert_color="black",
    gs_outer=None,
    fig=None,
    panel_label=None,
    show_pred=True,
    show_delta=False,
):
    assert kind in ["brain_stimulus", "brain_brain"]
    preproc_fun = preproc_onestim if kind == "brain_stimulus" else preproc_twostim

    n_methods = len(methods)
    lh_pick = int(lh_pick) if not np.isnan(lh_pick) else None
    rh_pick = int(rh_pick) if not np.isnan(rh_pick) else None
    coh_theory = get_theory(
        kind,
        ctfs,
        p.labels,
        src,
        model,
        lh_pick,
        rh_pick,
        gamma,
        average=True,
    )
    coh_theory_fit, coh_real_fit, consider_fit = preproc_fun(
        coh_theory, coh_real, coh_threshold
    )
    coh_theory_all, coh_real_all, consider_all = preproc_fun(coh_theory, coh_real)
    if show_pred:
        fm, intercept, slope, _, pred_all = fit_and_predict(
            coh_theory_fit, coh_real_fit, coh_theory_all, model
        )

    coh_theory = pred_all.reshape((n_methods, -1))
    coh_real = coh_real_all.reshape((n_methods, -1))
    coh_theory_mean = coh_theory.mean(axis=0)
    coh_real_mean = coh_real.mean(axis=0)

    # Create the subplot grid
    # Number of rows = number of methods + 1 (column labels)
    # Number of columns = 4 (method labels, theory, real data, comparison)
    if gs_outer is None:
        fig = plt.figure(figsize=(12, 3 * n_methods), layout="constrained")
        gs = fig.add_gridspec(
            nrows=n_methods + 2,
            ncols=4,
            width_ratios=[0.1, 1, 1, 1],
            height_ratios=[0.2, 1, 1, 1, 0.1],
        )
    else:
        assert fig is not None
        gs = gs_outer.subgridspec(
            nrows=n_methods + 2,
            ncols=4,
            width_ratios=[0.1, 1, 1, 1],
            height_ratios=[0.2, 1, 1, 1, 0.1],
        )

    # Add panel and column labels
    if panel_label is not None:
        ax_panel_label = fig.add_subplot(gs[0, 0])
        ax_panel_label.axis("off")
        add_label(ax_panel_label, panel_label, x=0)
    for i_label, label in enumerate(["Theory", "Real data", "Comparison"]):
        ax_label = fig.add_subplot(gs[0, 1 + i_label])
        draw_text(
            ax_label,
            label.replace("_", "-"),
            fontsize="x-large",
        )

    # Common limits for theoretical and real estimates
    if show_delta:
        lim_th = 1.05 * np.abs(coh_theory).max()
        lim_real = 1.05 * np.abs(coh_real).max()
        lim = max(lim_th, lim_real)
        lims = [-lim, 0, lim]
        cmap = "RdBu_r"
    else:
        lim_th = 1.05 * coh_theory.max()
        lim_real = 1.05 * coh_real.max()
        lim = max(lim_th, lim_real)
        thresh = coh_threshold if show_pred else 0.2 * lim
        lims = [0, thresh, lim]
        cmap = "viridis"

    # Plot the theoretical and real data estimates
    for i_method, method in enumerate(methods):
        test_idx = np.where(coh_real[i_method, :] > coh_threshold)[0]

        if show_delta:
            coh_method_theory = coh_theory[i_method, :] - coh_theory_mean
            coh_method_real = coh_real[i_method, :] - coh_real_mean
        else:
            coh_method_theory = coh_theory[i_method, :]
            coh_method_real = coh_real[i_method, :]

        corr = metric_fun(
            coh_method_theory[test_idx],
            coh_method_real[test_idx],
        )
        corr_all = metric_fun(coh_method_theory, coh_method_real)

        ax_label = fig.add_subplot(gs[1 + i_method, 0])
        draw_text(
            ax_label,
            method.replace("_", "-"),
            facecolor="#dddddd",
            rotation="vertical",
            fontsize="x-large",
        )

        ax1 = fig.add_subplot(gs[1 + i_method, 1])
        if kind == "brain_stimulus":
            plot_coh_onestim(
                coh_method_theory,
                src,
                p,
                lh_pick,
                rh_pick,
                ax=ax1,
                clim=dict(kind="value", lims=lims),
                vert_color=vert_color,
            )
        else:
            vmin, center, vmax = lims
            plot_imcoh_twostim(
                coh_method_theory,
                p,
                vmin=vmin,
                center=center if show_delta else None,
                vmax=vmax,
                ax=ax1,
            )

        ax2 = fig.add_subplot(gs[1 + i_method, 2])
        if kind == "brain_stimulus":
            plot_coh_onestim(
                coh_method_real,
                src,
                p,
                None,
                None,
                ax=ax2,
                colormap=cmap,
                clim=dict(kind="value", lims=lims),
                vert_color=vert_color,
            )
        else:
            vmin, center, vmax = lims
            plot_imcoh_twostim(
                coh_method_real,
                p,
                vmin=vmin,
                center=center if show_delta else None,
                vmax=vmax,
                ax=ax2,
            )

        xlim = [-lim if show_delta else 0, lim]
        ylim = [-lim if show_delta else 0, lim]
        ax3 = fig.add_subplot(gs[1 + i_method, 3])
        ax3.scatter(coh_method_theory, coh_method_real, c="grey")
        ax3.scatter(coh_method_theory[test_idx], coh_method_real[test_idx], c="red")
        ax3.axhline(coh_threshold, c="gray", ls=":")
        ax3.plot([-1, 1], [-1, 1], c="gray", ls="--")
        ax3.text(
            0.05,
            0.85,
            f"$r_{{fit}}$ = {corr:.2f}",
            color="red",
            fontsize="large",
            transform=ax3.transAxes,
        )
        ax3.text(
            0.05,
            0.7,
            f"$r_{{all}}$ = {corr_all:.2f}",
            color="grey",
            fontsize="large",
            transform=ax3.transAxes,
        )
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        if i_method != 2:
            ax3.set_xticks([])
        if i_method == 2:
            ax3.set_xlabel("Theory")
        ax3.set_ylabel("Real data")

    # Add a colorbar to the connectivity plots
    ax_cbar = fig.add_subplot(gs[-1, 1])
    if kind == "brain_stimulus":
        mne.viz.plot_brain_colorbar(
            ax_cbar,
            clim=dict(kind="value", lims=lims),
            colormap=cmap,
            transparent=False,
            orientation="horizontal",
            label="Brain-stimulus coherence",
        )
        ax_cbar.set_xticks(np.arange(0, lim, 0.2))
    else:
        vmin, _, vmax = lims
        sm = ScalarMappable(Normalize(vmin, vmax), cmap="rocket")
        fig.colorbar(
            sm, cax=ax_cbar, orientation="horizontal", label="Brain-brain ImCoh"
        )
        ax_cbar.set_xticks(np.arange(0, lim, 0.1))

    if gs_outer is None:
        return fig
