import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from PIL import Image

from ctfeval.config import paths, params
from ctfeval.datasets import rift_subfolder
from ctfeval.io import init_subject
from ctfeval.rift.metrics import METRICS
from ctfeval.rift.viz import plot_fit_extended
from ctfeval.utils import vertno_to_index
from ctfeval.viz import (
    set_plot_style,
    add_label,
    fsaverage_brain,
    make_cropped_screenshot,
)

set_plot_style()


def main_figure_onestim(
    best_fit_raw,
    best_fit_delta,
    coh_stim_ga,
    times,
    stim,
    ssvep_left,
    ctf_array,
    coh_real,
    coh_threshold,
    methods,
    src,
    p,
    leadfield_mean,
    info_plot,
    metric_name,
    target,
    ssvep_tmin=0.4,
    ssvep_tmax=0.6,
):
    # Figure layout
    fig = plt.figure(figsize=(10, 10.5), layout="constrained")
    gs = fig.add_gridspec(nrows=3, height_ratios=[1.1, 0.05, 2])
    sgs = gs[0].subgridspec(
        nrows=4,
        ncols=8,
        height_ratios=[0.5, 0.5, 0.5, 0.5],
        width_ratios=[0.575, 0.575, 0.05, 0.6, 0.05, 0.05, 0.7, 0.2],
    )

    # Panel A: rationale
    ax = fig.add_subplot(sgs[:2, 0])
    img = Image.open(paths.assets / "rift-onestim-setup.png")
    ax.imshow(img)
    ax.axis("off")
    add_label(ax, "A")

    # Panel A: search space = V1 / pericalcarine
    ax_brain = fig.add_subplot(sgs[:2, 1])
    brain = fsaverage_brain(hemi="lh", views="med")
    brain.add_label(p["pericalcarine-lh"], color="red", borders=False)
    make_cropped_screenshot(brain, ax=ax_brain)

    # Panel A: stimulus and response waveforms
    ax_waveform = fig.add_subplot(sgs[2:, :2])
    time_idx = np.logical_and(times >= ssvep_tmin, times <= ssvep_tmax)

    ax_waveform.plot(times[time_idx], stim[time_idx], c="grey", label="stimulus")
    ax_waveform.plot(
        times[time_idx], ssvep_left[time_idx], c="tab:red", label="brain response"
    )
    ax_waveform.set_xlim([ssvep_tmin, ssvep_tmax])
    ax_waveform.set_xticks(np.arange(ssvep_tmin, ssvep_tmax + 0.01, 0.05))
    ax_waveform.set_yticks([])
    ax_waveform.set_xlabel("Time (s)")
    ax_waveform.set_ylabel("Amplitude (a.u.)")
    ax_waveform.legend(
        ncols=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.55, 1.05),
        bbox_transform=ax_waveform.transAxes,
    )

    # Panel B: sensor-space brain-stimulus coherence
    ax_sensor_space = fig.add_subplot(sgs[:2, 3])
    im, _ = mne.viz.plot_topomap(
        coh_stim_ga, info_plot, axes=ax_sensor_space, show=False, vlim=(0, 0.7)
    )
    add_label(ax_sensor_space, "B")

    ax_cbar = fig.add_subplot(sgs[:2, 4])
    fig.colorbar(im, cax=ax_cbar)
    ax_cbar.set_ylabel("Coherence")

    # Panel C: leadfield of the best CTF fit
    best_ctf_fit = best_fit_raw[best_fit_raw.model == "CTF"]

    # NOTE: These params are also used in the extended fit
    lh_seed = int(best_ctf_fit.lh_seed.iloc[0])
    rh_seed = int(best_ctf_fit.rh_seed.iloc[0])
    gamma = int(best_ctf_fit.gamma.iloc[0])

    lh_idx = vertno_to_index(src, "lh", lh_seed)
    lh_leadfield = np.squeeze(leadfield_mean[:, lh_idx])
    rh_idx = vertno_to_index(src, "rh", rh_seed)
    rh_leadfield = np.squeeze(leadfield_mean[:, rh_idx])

    ax_leadfield = fig.add_subplot(sgs[2:, 3])
    im, _ = mne.viz.plot_topomap(
        lh_leadfield + rh_leadfield, info_plot, axes=ax_leadfield, show=False
    )
    add_label(ax_leadfield, "C")

    ax_leadfield_cbar = fig.add_subplot(sgs[2:, 4])
    fig.colorbar(im, cax=ax_leadfield_cbar)
    ax_leadfield_cbar.set_ylabel("Lead field")

    # Panel D: model comparison
    metric_full_raw = f"{metric_name}_raw_{target}"
    metric_full_delta = f"{metric_name}_delta_{target}"
    best_fit_raw = best_fit_raw.sort_values(metric_full_raw)
    best_fit_delta = best_fit_delta[~pd.isna(best_fit_delta[metric_full_delta])]

    # raw
    ax_raw = fig.add_subplot(sgs[:, 6])
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
    ax_raw.xaxis.set_label_coords(0.65, -0.1)
    add_label(ax_raw, "D", x=-0.2, y=1.06)

    # delta
    ax_delta = fig.add_subplot(sgs[:, 7])
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

    # Panel E: extended fit for CTF
    metric_fun = METRICS[metric_name]
    plot_fit_extended(
        "brain_stimulus",
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
        panel_label="E",
        show_pred=True,
        show_delta=False,
    )

    # Save the result
    fig.savefig(paths.figures / "fig8_rift_onestim.png", dpi=300, bbox_inches="tight")


def main():
    # Paths
    tag_folder = paths.rift / rift_subfolder(
        params.rift_onestim.tagging_type, params.rift_onestim.random_phases
    )
    theory_path = tag_folder / "theory"
    brain_stim_path = tag_folder / "brain_stimulus"

    # Load common resources
    info_plot = mne.io.read_info(paths.rift / "info" / "plot-info.fif")
    fwd, _, _, p = init_subject(parcellation="DK")
    src = fwd["src"]

    # Load average leadfield for plotting
    leadfield_mean = np.load(paths.rift / "headmodels" / "leadfield_mean.npy")

    # Load sensor-space brain-stimulus coherence
    coh_path = tag_folder / "sensor_space" / "grand_average_onestim_abscoh_stim1.npy"
    coh_stim_ga = np.load(coh_path)

    # Load the grand-average SSVEP time course in V1
    roi_path = tag_folder / "roi_space"

    times = np.load(roi_path / "times.npy")
    stim = np.load(roi_path / "stim.npy")
    ssvep_left = np.load(roi_path / "v1_left_ga.npy")

    # 20 Hz high-pass for illustration purposes
    sfreq = 1.0 / (times[1] - times[0])
    hp = 20
    b, a = butter(5, 2 * hp / sfreq, btype="high")

    ssvep_left_filt = filtfilt(b, a, ssvep_left)

    # Normalize to have zero mean and the same variance
    stim_norm = (stim - np.mean(stim)) / np.std(stim)
    ssvep_left_norm = (ssvep_left_filt - np.mean(ssvep_left_filt)) / np.std(
        ssvep_left_filt
    )

    # Load noise floor
    with np.load(theory_path / "noise_floor_coh_onestim.npz") as data:
        coh_threshold = data["conn_threshold"]

    methods = ["mean", "mean_flip", "centroid"]

    # Load real data results (brain-stimulus coherence, 1 stimulus)
    abscoh_1stim = np.load(brain_stim_path / "onestim_avg_abscoh_stim.npy")

    # Load CTFs
    ctf_array = np.load(paths.rift / "ctf_array.npy")

    # Load results of model comparison
    best_fit_raw = pd.read_csv(
        brain_stim_path
        / f"best_results_{params.rift_onestim.metric}_raw_{params.rift_onestim.target}.csv"
    )
    best_fit_delta = pd.read_csv(
        brain_stim_path
        / f"best_results_{params.rift_onestim.metric}_delta_{params.rift_onestim.target}.csv"
    )
    for df in [best_fit_raw, best_fit_delta]:
        df.replace(
            to_replace={
                "model": {"no_leakage": "No RFS", "distance": "Distance", "ctf": "CTF"}
            },
            inplace=True,
        )

    # Prepare the figure
    main_figure_onestim(
        best_fit_raw,
        best_fit_delta,
        coh_stim_ga,
        times,
        stim_norm,
        ssvep_left_norm,
        ctf_array,
        abscoh_1stim,
        coh_threshold,
        methods,
        src,
        p,
        leadfield_mean,
        info_plot,
        params.rift_onestim.metric,
        params.rift_onestim.target,
    )


if __name__ == "__main__":
    main()
