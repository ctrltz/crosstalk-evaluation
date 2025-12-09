import numpy as np
import mne

from mne.filter import construct_iir_filter
from scipy.signal import butter, filtfilt
from sklearn.model_selection import KFold
from tqdm import tqdm

from ctfeval.log import logger
from ctfeval.prepare import setup_inverse_operator


def construct_iir_params(raw, fmin, fmax, order=2):
    iir_params = dict(order=order, ftype="butter", output="sos")
    iir_params = construct_iir_filter(
        iir_params=iir_params,
        f_pass=[fmin, fmax],
        f_stop=None,
        sfreq=raw.info["sfreq"],
        btype="bandpass",
        return_copy=False,
    )


def run_cv(raw, fwd, lambda2, method, kf, all_chans):
    """
    Spatial cross-validation (k-fold):
     - apply inverse method to the training subset of channels
     - evaluate predictions for the validation subset of channels
    """
    X_rec = np.zeros_like(raw.get_data())
    X_res = np.zeros_like(raw.get_data())
    for fit_indices, val_indices in kf.split(all_chans):
        fit_chans = list(all_chans[fit_indices])
        val_chans = list(all_chans[val_indices])

        # Fit
        fwd_fit = mne.pick_channels_forward(fwd, fit_chans, ordered=True, verbose=False)
        raw_fit = raw.copy().pick(fit_chans)

        # Prepare and fit the inverse operator
        inv_fit = setup_inverse_operator(fwd_fit, raw_fit.info)
        stc = mne.minimum_norm.apply_inverse_raw(
            raw_fit, inv_fit, lambda2=lambda2, method=method, verbose=False
        )

        # Validate
        fwd_val = mne.pick_channels_forward(fwd, val_chans, ordered=True, verbose=False)
        raw_val = raw.copy().pick(val_chans)
        raw_rec = mne.apply_forward_raw(fwd_val, stc, raw_val.info, verbose=False)

        # Evaluate
        X_val = raw_val.get_data()
        X_rec[val_indices, :] = raw_rec.get_data()
        X_res[val_indices, :] = X_val - X_rec[val_indices, :]

    return X_rec, X_res


def infer_noise_level_cv(
    raw,
    fwd,
    tmin=0.0,
    tmax=None,
    fmin=None,
    fmax=None,
    method="eLORETA",
    lambdas=np.logspace(-4, 1, num=15),
    n_cv_splits=5,
    random_state=None,
):
    # Initialize the k-fold splitter
    all_chans = np.array(raw.ch_names)
    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)

    # Crop the provided raw data and apply average reference
    raw_cv = raw.copy().crop(tmin=tmin, tmax=tmax).apply_proj()

    # Calculate total power
    data_orig = raw_cv.get_data()
    filtering_needed = fmin is not None and fmax is not None
    if filtering_needed:
        b, a = butter(
            2, 2 * np.array([fmin, fmax]) / raw.info["sfreq"], btype="bandpass"
        )
        data_orig = filtfilt(b, a, data_orig)
    power_total = np.diag(data_orig @ data_orig.T)

    # Store total power and power of the residuals
    n_lambdas = lambdas.size
    n_chans = all_chans.size
    power_res = np.zeros((n_lambdas, n_chans))
    for i_lambda, lambda2 in enumerate(tqdm(lambdas)):
        _, X_res = run_cv(raw_cv, fwd, lambda2, method, kf, all_chans)

        # Filter
        if filtering_needed:
            X_res = filtfilt(b, a, X_res)

        power_res[i_lambda, :] = np.diag(X_res @ X_res.T)

    ratios = power_res.mean(axis=1) / power_total.mean()
    noise_level = ratios.min()
    best_lambda = lambdas[ratios.argmin()]

    return noise_level, best_lambda


def infer_noise_level_amplifier(raw, fmin, fmax, famp, fres=0.5, overlap=0.5):
    """
    Calculate the ratio between the power of amplifier noise (evaluated as power
    at famp Hz frequency) to the total power in fmin-fmax Hz range, averaged
    over channels
    """
    n_fft = int(raw.info["sfreq"] / fres)
    n_overlap = int(overlap * n_fft)
    spec = raw.compute_psd(
        method="welch", n_per_seg=n_fft, n_overlap=n_overlap, n_fft=n_fft
    )
    psd, freqs = spec.get_data(return_freqs=True)
    psd_avg = psd.mean(axis=0)

    amp_noise = spec.get_data(fmin=famp, fmax=famp).mean()
    band_power = spec.get_data(fmin=fmin, fmax=fmax).mean()
    noise_level = amp_noise / band_power

    return noise_level, amp_noise, psd_avg, freqs


def infer_sensor_space_snr(raw, fmin, fmax, fres=0.5, overlap=0.5):
    """
    Calculate the SNR similar to SSD (target band vs. side bands) based on the averaged
    spectra of all channels
    """
    n_fft = int(raw.info["sfreq"] / fres)
    n_overlap = int(overlap * n_fft)
    spec = raw.compute_psd(
        method="welch", n_fft=n_fft, n_per_seg=n_fft, n_overlap=n_overlap
    )
    psd, freqs = spec.get_data(return_freqs=True)
    psd_avg = psd.mean(axis=0)

    alpha_freqs = np.logical_and(freqs >= fmin, freqs <= fmax)
    side_freqs = np.logical_or(
        np.logical_and(freqs >= (fmin - 3), freqs <= (fmin - 1)),
        np.logical_and(freqs >= (fmax + 1), freqs <= (fmax + 3)),
    )
    logger.info(f"Sensor-space SNR - alpha freqs: [{fmin}, {fmax}] Hz")
    logger.info(
        f"Sensor-space SNR - side freqs: [{fmin-3}, {fmin-1}] & [{fmax+1}, {fmax+3}] Hz"
    )

    snr = psd_avg[alpha_freqs].mean() / psd_avg[side_freqs].mean()
    return snr, psd_avg, freqs


def infer_source_power(
    raw, method, fmin=None, fmax=None, fwd=None, inv=None, reg=0.05, cov_tstep=2.0
):
    # Parameters for filtering
    raw_cov = raw.copy()
    filtering_needed = fmin is not None and fmax is not None
    if filtering_needed:
        iir_params = construct_iir_params(raw, fmin, fmax)
        raw_cov = raw_cov.filter(
            l_freq=fmin, h_freq=fmax, method="iir", iir_params=iir_params
        )

    # Sensor space covariance
    sensor_cov = mne.compute_raw_covariance(raw_cov, tstep=cov_tstep)

    # Project to source space
    if method in ["dSPM", "MNE", "sLORETA", "eLORETA"]:
        assert inv is not None
        source_power = mne.minimum_norm.apply_inverse_cov(
            sensor_cov, raw.info, inv, lambda2=reg, method=method
        )
    elif method == "LCMV":
        assert fwd is not None
        lcmv_filters = mne.beamformer.make_lcmv(raw.info, fwd, sensor_cov, reg=reg)
        source_power = mne.beamformer.apply_lcmv_cov(sensor_cov, lcmv_filters)
    else:
        raise NotImplementedError("Unsupported method {method}")

    return source_power.data
