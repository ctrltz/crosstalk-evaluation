import mne
import numpy as np

from scipy.signal import butter

from ctfeval.log import logger


def prepare_forward(
    subjects_dir,
    subject,
    src,
    info,
    meg=True,
    eeg=True,
    mindist=5.0,
    subfolder="bem",
    spacing="oct6",
    bem_file="fsaverage-5120-5120-5120-bem-sol.fif",
    trans="fsaverage",
    save=False,
    overwrite=False,
):
    # Load and return if exists
    bem = subjects_dir / subject / subfolder / bem_file
    fwd_path = subjects_dir / subject / subfolder / f"{subject}-{spacing}-fwd.fif"

    # Compute the forward solution
    logger.info(f"Creating a forward model for {spacing} spacing, {meg=}, {eeg=}")
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        meg=meg,
        eeg=eeg,
        mindist=mindist,
        n_jobs=None,
        verbose=True,
    )
    if save:
        logger.info(f"Saving the forward model to {fwd_path}")
        mne.write_forward_solution(fwd_path, fwd, overwrite=overwrite)

    return fwd


def prepare_source_space(
    subjects_dir,
    subject,
    subfolder="bem",
    spacing="oct6",
    plot_src=False,
    save=False,
    overwrite=False,
):
    # Load and return if exists
    src_path = subjects_dir / subject / subfolder / f"{subject}-{spacing}-src.fif"

    # Setup the source space
    logger.info(f"Creating a source space for {spacing} spacing")
    src = mne.setup_source_space(
        subject, spacing=spacing, add_dist=False, subjects_dir=subjects_dir
    )
    if save:
        logger.info(f"Saving the source space to {src_path}")
        mne.write_source_spaces(src_path, src, overwrite=overwrite)

    # Plot the source space
    if plot_src:
        plot_bem_kwargs = dict(
            subject=subject,
            subjects_dir=subjects_dir,
            brain_surfaces="white",
            orientation="coronal",
            slices=[50, 100, 150, 200],
        )
        fig = mne.viz.plot_bem(src=src, **plot_bem_kwargs)
        return src, fig

    return src


def setup_inverse_operator(fwd, info):
    # Use identity as the noise covariance matrix
    noise_cov = mne.make_ad_hoc_cov(info, std=1.0)

    inv = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, fixed=True, depth=None
    )

    return inv


def prepare_filter(sfreq, fmin, fmax, order=2):
    return butter(order, 2 * np.array([fmin, fmax]) / sfreq, btype="bandpass")


def interpolate_missing_channels(raw, full_info):
    # Restore the missing channels and marked them as bad
    missing_names = list(set(full_info.ch_names) - set(raw.info["ch_names"]))
    if not missing_names:
        return raw.reorder_channels(full_info.ch_names)

    logger.info(
        f"Interpolating {len(missing_names)} missing channels: {', '.join(missing_names)}"
    )
    missing_data = np.zeros((len(missing_names), raw.n_times))
    missing_info = mne.create_info(
        ch_names=missing_names, sfreq=raw.info["sfreq"], ch_types="eeg"
    )
    missing_info["bads"].extend(missing_names)

    # Add the missing channels to the provided raw
    missing_raw = mne.io.RawArray(
        data=missing_data,
        info=missing_info,
        first_samp=raw.first_samp,
    )
    raw.add_channels([missing_raw])

    # Fill in the channel positions
    raw.info.set_montage(full_info.get_montage())

    # Interpolate the missing channels and match the order of the provided Info
    raw.interpolate_bads()
    return raw.reorder_channels(full_info.ch_names)
