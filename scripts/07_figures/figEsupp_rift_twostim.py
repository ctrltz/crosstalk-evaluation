import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from PIL import Image

from ctfeval.config import paths, params
from ctfeval.datasets import rift_subfolder
from ctfeval.io import init_subject
from ctfeval.rift.metrics import METRICS
from ctfeval.rift.viz import plot_fit_extended
from ctfeval.viz import add_label, set_plot_style
from ctfeval.utils import vertno_to_index

set_plot_style()


def supp_figure_twostim(
    best_fit_raw,
    best_fit_delta,
    times,
    tag1,
    tag2,
    coh_real,
    coh_threshold,
    methods,
    src,
    p,
    ctf_array,
    leadfield_mean,
    info_plot,
    metric_name,
    target,
    ssvep_tmin=0.4,
    ssvep_tmax=0.5,
):
    # Figure layout
    fig = plt.figure(figsize=(10, 10), layout="constrained")
    gs = fig.add_gridspec(nrows=3, height_ratios=[0.6, 0.05, 2.5])
    sgs = gs[0].subgridspec(
        nrows=1,
        ncols=7,
        width_ratios=[0.7, 0.7, 0.5, 0.05, 0.05, 0.6, 0.2],
    )

    # Panel A: rationale + stimulus' time courses
    ax = fig.add_subplot(sgs[0])
    img = Image.open(paths.assets / "rift-twostim-setup.png")
    ax.imshow(img)
    ax.axis("off")
    add_label(ax, "A")

    # NOTE: labels were set based on quote from the paper - "separately
    # for the 0° (left stimulus) and 90° (right stimulus) cases"
    ax = fig.add_subplot(sgs[1])
    mask = np.logical_and(times >= ssvep_tmin, times <= ssvep_tmax)
    ax.plot(times[mask], tag2[0, mask], c="black", label="left")
    ax.plot(times[mask], tag1[0, mask], c="grey", label="right")
    ax.set_xlabel("Time (s)")
    ax.set_xticks(np.arange(ssvep_tmin, ssvep_tmax + 0.01, 0.05))
    ax.set_ylabel("Luminance (a.u.)")
    ax.set_ylim([0, 1.5])
    ax.legend(loc="upper center", frameon=False, ncols=2)

    # Panel B: leadfield of the best CTF fit
    best_ctf_fit = best_fit_raw[best_fit_raw.model == "CTF"]

    # NOTE: These params are also used in the extended fit
    lh_seed = int(best_ctf_fit.lh_seed.iloc[0])
    rh_seed = int(best_ctf_fit.rh_seed.iloc[0])
    gamma = int(best_ctf_fit.gamma.iloc[0])

    lh_idx = vertno_to_index(src, "lh", lh_seed)
    lh_leadfield = np.squeeze(leadfield_mean[:, lh_idx])
    rh_idx = vertno_to_index(src, "rh", rh_seed)
    rh_leadfield = np.squeeze(leadfield_mean[:, rh_idx])

    ax_leadfield = fig.add_subplot(sgs[2])
    im, _ = mne.viz.plot_topomap(
        lh_leadfield + rh_leadfield, info_plot, axes=ax_leadfield, show=False
    )
    add_label(ax_leadfield, "B")

    ax_leadfield_cbar = fig.add_subplot(sgs[3])
    fig.colorbar(im, cax=ax_leadfield_cbar)
    ax_leadfield_cbar.set_ylabel("Lead field")

    # Panel C: model comparison
    metric_full_raw = f"{metric_name}_raw_{target}"
    metric_full_delta = f"{metric_name}_delta_{target}"
    best_fit_raw = best_fit_raw.sort_values(metric_full_raw)
    best_fit_delta = best_fit_delta[~pd.isna(best_fit_delta[metric_full_delta])]

    # raw
    ax_raw = fig.add_subplot(sgs[5])
    best_fit_raw.plot.bar(
        x="model",
        y=metric_full_raw,
        xlabel="Model",
        ylabel="Correlation",
        ylim=[0, 1],
        ax=ax_raw,
        rot=0,
        legend=False,
    )
    ax_raw.set_title("raw")
    ax_raw.xaxis.set_label_coords(0.65, -0.25)
    add_label(ax_raw, "C", x=-0.3, y=1.15)

    # delta
    ax_delta = fig.add_subplot(sgs[6])
    best_fit_delta.plot.bar(
        x="model",
        y=metric_full_delta,
        xlabel="",
        ylabel="",
        ylim=[0, 1],
        ax=ax_delta,
        rot=0,
        legend=False,
    )
    ax_delta.set_yticklabels([])
    ax_delta.set_title("delta")

    # Panel D: extended fit for CTF
    metric_fun = METRICS[metric_name]
    plot_fit_extended(
        "brain_brain",
        ctf_array,
        src,
        p,
        np.array(methods),
        coh_real,
        coh_threshold,
        metric_fun,
        "ctf",
        lh_seed,
        rh_seed,
        gamma,
        gs_outer=gs[2],
        fig=fig,
        panel_label="D",
        show_pred=True,
        show_delta=False,
    )

    # Save the result
    fig.savefig(
        paths.figures / "figEsupp_rift_twostim.png", dpi=300, bbox_inches="tight"
    )


def main():
    # Paths
    tag_folder = paths.rift / rift_subfolder(
        params.rift_twostim.tagging_type, params.rift_twostim.random_phases
    )
    theory_path = tag_folder / "theory"
    brain_brain_path = tag_folder / "brain_brain"
    preproc_path = tag_folder / "preproc"

    # Load common resources
    info_plot = mne.io.read_info(paths.rift / "info" / "plot-info.fif")
    fwd, _, _, p = init_subject(parcellation="DK")
    src = fwd["src"]

    # Load average leadfield for plotting
    leadfield_mean = np.load(paths.rift / "headmodels" / "leadfield_mean.npy")

    # Load the stimulation signal from an exemplary subject
    epochs = mne.read_epochs(preproc_path / "sub001_data_twostim-epo.fif")
    tag1 = np.load(preproc_path / "sub001_data_twostim_tag1.npy")
    tag2 = np.load(preproc_path / "sub001_data_twostim_tag2.npy")

    # Load noise floor
    with np.load(theory_path / "noise_floor_imcoh_twostim.npz") as data:
        coh_threshold = data["conn_threshold"]

    # Load real data results (brain-brain ImCoh, 2 stimuli)
    absimcoh_2stim = np.load(brain_brain_path / "twostim_avg_absimcoh.npy")

    # Load CTFs
    ctf_array = np.load(paths.rift / "ctf_array.npy")

    # Load results of model comparison
    best_fit_raw = pd.read_csv(
        brain_brain_path
        / f"best_results_{params.rift_twostim.metric}_raw_{params.rift_twostim.target}.csv"
    )
    best_fit_delta = pd.read_csv(
        brain_brain_path
        / f"best_results_{params.rift_twostim.metric}_delta_{params.rift_twostim.target}.csv"
    )
    for df in [best_fit_raw, best_fit_delta]:
        df.replace(
            to_replace={
                "model": {"no_leakage": "No RFS", "distance": "Distance", "ctf": "CTF"}
            },
            inplace=True,
        )

    supp_figure_twostim(
        best_fit_raw=best_fit_raw,
        best_fit_delta=best_fit_delta,
        times=epochs.times,
        tag1=tag1,
        tag2=tag2,
        coh_real=absimcoh_2stim,
        coh_threshold=coh_threshold,
        methods=["mean", "mean_flip", "centroid"],
        src=src,
        p=p,
        ctf_array=ctf_array,
        leadfield_mean=leadfield_mean,
        info_plot=info_plot,
        metric_name=params.rift_onestim.metric,
        target=params.rift_onestim.target,
    )


if __name__ == "__main__":
    main()
