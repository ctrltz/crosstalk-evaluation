import matplotlib.pyplot as plt
import mne
import numpy as np

from scipy.signal import filtfilt

from ctfeval.config import params, paths
from ctfeval.extraction import (
    apply_inverse_with_weights,
    extract_label_time_course_with_weights,
)
from ctfeval.io import init_subject
from ctfeval.log import logger
from ctfeval.prepare import prepare_filter
from ctfeval.simulations import simulation_activity_reconstruction
from ctfeval.utils import data2stc
from ctfeval.viz import set_plot_style, make_cropped_screenshot

set_plot_style()


# Adjust the colors of the cortex to increase contrast with added elements
CORTEX = dict(colormap="Greys", vmin=-2, vmax=5)


def panel_workflow():
    """
    Workflow of the simulations
    """
    logger.info("Creating figures for simulation workflow")
    color_rec = "black"
    color_gt = "green"

    fwd, inv, lemon_info, p = init_subject(parcellation="DK")
    src = fwd["src"]
    target_label = p["parsopercularis-rh"]

    logger.info("Preparing the simulation")
    # Define the simulation
    sim = simulation_activity_reconstruction(
        fwd=fwd,
        labels=p.labels,
        n_noise_dipoles=params.n_noise_dipoles,
        target_snr=params.target_snr,
        snr_mode="global",
        std=1.0,
        source_type="point",
        patch_area_cm2=None,
        n_connections=0,
        mean_coherence=0.0,
        std_coherence=0.0,
        fmin=params.fmin,
        fmax=params.fmax,
        random_state=params.seed,
    )

    # Simulate the data
    sc = sim.simulate(
        sfreq=params.sfreq,
        duration=params.duration,
        fwd=fwd,
        snr_global=params.target_snr,
        snr_params=dict(fmin=params.fmin, fmax=params.fmax),
        random_state=params.seed,
    )

    # Project to sensor space
    raw = sc.to_raw(fwd, lemon_info, sensor_noise_level=params.sensor_noise_level)
    raw.set_eeg_reference(projection=True)

    # Prepare the ground truth and reconstructed time series
    b, a = prepare_filter(sc.sfreq, params.fmin, params.fmax)

    stc = apply_inverse_with_weights(raw, fwd, inv, "eLORETA", label=target_label)
    label_tc = extract_label_time_course_with_weights(
        stc, target_label, src, "mean_flip", "fsaverage", paths.subjects_dir
    )
    label_tc = filtfilt(b, a, label_tc)

    gt = sc[target_label.name].waveform
    rec = np.squeeze(label_tc)

    # Plot the source configuration
    logger.info("Plotting the source configuration")
    fig, ax = plt.subplots()
    brain = sc.plot(
        subject="fsaverage",
        subjects_dir=paths.subjects_dir,
        hemi="rh",
        cortex=CORTEX,
        colors=dict(noise="#555555", point=color_gt),
        size=(1200, 1200),
        volume_options=1,
    )
    brain.add_annotation(p.fs_name, color=color_rec)
    make_cropped_screenshot(brain, ax=ax, close=True)
    fig.savefig(paths.figure_components / "fig2_sc.png", bbox_inches="tight")
    plt.close(fig)

    # Plot the sensor configuration
    logger.info("Plotting the sensor configuration")
    fig, ax = plt.subplots(figsize=(4, 4))
    lemon_info.plot_sensors(sphere="eeglab", axes=ax, show=False)
    fig.savefig(
        paths.figure_components / "fig2_info.svg", bbox_inches="tight", transparent=True
    )
    plt.close(fig)

    # Plot the waveforms
    logger.info("Plotting the waveforms")
    seconds_to_plot = 1
    samples_to_plot = seconds_to_plot * sc.sfreq
    times = sc.times[:samples_to_plot]

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 3))
    ax.plot(times, rec[:samples_to_plot], color=color_rec)
    ax.axis("off")
    fig.savefig(
        paths.figure_components / "fig2_rec.svg", bbox_inches="tight", transparent=True
    )
    plt.close(fig)

    fig, ax = plt.subplots(layout="constrained", figsize=(4, 3))
    ax.plot(times, gt[:samples_to_plot], color=color_gt)
    ax.axis("off")
    fig.savefig(
        paths.figure_components / "fig2_gt.svg", bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def panel_experiments():
    """
    Experiments
    """
    logger.info("Creating figures for experiments")
    fwd, _, lemon_info, p = init_subject(parcellation="DK")
    src = fwd["src"]

    # Experiment 1. Source variance
    logger.info("Experiment 1 - source variance")
    var_filename = paths.precomputed / "var_oct6_EC_like.npy"
    source_var = np.load(var_filename)
    source_var /= np.max(source_var)
    std = data2stc(source_var, src)

    brain = std.plot(
        subject="fsaverage",
        subjects_dir=paths.subjects_dir,
        surface="inflated",
        hemi="rh",
        views="lat",
        view_layout="horizontal",
        background="w",
        cortex=CORTEX,
        clim=dict(kind="value", lims=[0, 0.5, 1.0]),
        colormap="Reds",
        transparent=False,
        colorbar=False,
    )
    fig, ax = plt.subplots()
    make_cropped_screenshot(brain, ax=ax)
    fig.savefig(paths.figure_components / "fig2_exp1.png", bbox_inches="tight")

    # Experiment 2. Source size
    # Place sources of different size for illustration
    logger.info("Experiment 2 - source size")
    vertnos = [154301, 82804, 82213, 158184]
    areas = [0, 2, 4, 8]
    fill_color = "#333333"

    Brain = mne.viz.get_brain_class()
    brain = Brain(
        subject="fsaverage",
        hemi="rh",
        surf="inflated",
        cortex=CORTEX,
        background="w",
        subjects_dir=paths.subjects_dir,
        views="lat",
    )
    for vertno, area in zip(vertnos, areas):
        if not area:
            brain.add_foci(
                vertno, coords_as_verts=True, color=fill_color, scale_factor=0.5
            )
            continue

        extent_mm = np.sqrt(area * 100 / np.pi)
        label = mne.grow_labels(
            "fsaverage", vertno, extent_mm, "rh", paths.subjects_dir
        )[0]
        brain.add_label(label, color=fill_color)

    fig, ax = plt.subplots()
    make_cropped_screenshot(brain, ax=ax)
    fig.savefig(paths.figure_components / "fig2_exp2.png", bbox_inches="tight")

    # Experiment 3. Connectivity
    logger.info("Experiment 3 - connectivity")

    # All sources have the same size now
    vertnos = [154301, 82804, 82213, 158184]
    area = 2
    fill_color = "#333333"

    Brain = mne.viz.get_brain_class()
    brain = Brain(
        subject="fsaverage",
        hemi="rh",
        surf="inflated",
        cortex=CORTEX,
        background="w",
        subjects_dir=paths.subjects_dir,
        views="lat",
    )
    for vertno in vertnos:
        extent_mm = np.sqrt(area * 100 / np.pi)
        label = mne.grow_labels(
            "fsaverage", vertno, extent_mm, "rh", paths.subjects_dir
        )[0]
        brain.add_label(label, color=fill_color)

    fig, ax = plt.subplots()
    make_cropped_screenshot(brain, ax=ax)
    fig.savefig(paths.figure_components / "fig2_exp3.png", bbox_inches="tight")

    # Experiment 4a. 1/f noise
    logger.info("Experiment 4a - 1/f noise")

    # Show the levels with different shades of red
    colors = ["#fcbba1", "#ef3b2c", "#67000d"]

    # Define the simulation
    sim = simulation_activity_reconstruction(
        fwd=fwd,
        labels=p.labels,
        n_noise_dipoles=params.n_noise_dipoles,
        target_snr=params.target_snr,
        snr_mode="global",
        std=1.0,
        source_type="point",
        patch_area_cm2=None,
        n_connections=0,
        mean_coherence=0.0,
        std_coherence=0.0,
        fmin=params.fmin,
        fmax=params.fmax,
        random_state=params.seed,
    )

    fig, ax = plt.subplots(figsize=(2.5, 2))
    for snr, color in zip([0.3, 1, 3], colors):
        # Simulate the data
        sc = sim.simulate(
            sfreq=params.sfreq,
            duration=params.duration,
            fwd=fwd,
            snr_global=snr,
            snr_params=dict(fmin=params.fmin, fmax=params.fmax),
            random_state=params.seed,
        )

        # Project to sensor space
        raw = sc.to_raw(fwd, lemon_info, sensor_noise_level=params.sensor_noise_level)
        spec = raw.compute_psd(
            fmin=3,
            fmax=25,
            method="welch",
            n_fft=2 * sc.sfreq,
            n_overlap=sc.sfreq,
            n_per_seg=2 * sc.sfreq,
        )
        psd, freqs = spec.get_data(return_freqs=True)
        ax.plot(np.log10(freqs), 10 * np.log10(psd.mean(axis=0)), color=color, lw=1.5)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("log(PSD)")
    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(18)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(paths.figure_components / "fig2_exp4a.svg", bbox_inches="tight")

    # Experiment 4b. Sensor noise
    logger.info("Experiment 4b - sensor noise")

    # Define the simulation
    sim = simulation_activity_reconstruction(
        fwd=fwd,
        labels=p.labels,
        n_noise_dipoles=params.n_noise_dipoles,
        target_snr=params.target_snr,
        snr_mode="global",
        std=1.0,
        source_type="point",
        patch_area_cm2=None,
        n_connections=0,
        mean_coherence=0.0,
        std_coherence=0.0,
        fmin=params.fmin,
        fmax=params.fmax,
        random_state=params.seed,
    )

    # Simulate the data
    sc = sim.simulate(
        sfreq=params.sfreq,
        duration=params.duration,
        fwd=fwd,
        snr_global=params.target_snr,
        snr_params=dict(fmin=params.fmin, fmax=params.fmax),
        random_state=params.seed,
    )

    fig, ax = plt.subplots(figsize=(2.5, 2))
    for sensor_noise_level, color in zip([0.01, 0.1, 0.25], colors):
        # Project to sensor space
        raw = sc.to_raw(fwd, lemon_info, sensor_noise_level=sensor_noise_level)
        spec = raw.compute_psd(
            fmin=3,
            fmax=25,
            method="welch",
            n_fft=2 * sc.sfreq,
            n_overlap=sc.sfreq,
            n_per_seg=2 * sc.sfreq,
        )
        psd, freqs = spec.get_data(return_freqs=True)
        ax.plot(np.log10(freqs), 10 * np.log10(psd.mean(axis=0)), color=color)

    ax.set_xlabel("Frequency")
    ax.set_ylabel("log(PSD)")
    for item in [ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(18)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(paths.figure_components / "fig2_exp4b.svg", bbox_inches="tight")


def main():
    panel_workflow()
    panel_experiments()


if __name__ == "__main__":
    main()
