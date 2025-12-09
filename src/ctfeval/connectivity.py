import numpy as np

from scipy.signal import hilbert, welch, csd
from tqdm import tqdm

from meegsim.waveform import white_noise


def data2cs_hilbert(data):
    if data.ndim < 2:
        data = data[np.newaxis, :]

    data_hb = np.asmatrix(hilbert(data))
    cs = np.conjugate(data_hb @ data_hb.H)
    return np.asarray(cs)


def cpsd_welch(X, sfreq, nfft, nperseg, noverlap, window="hann", average="mean"):
    nchan = X.shape[0]
    nbins = int(nfft / 2 + 1)

    cs = np.empty(shape=(nbins, nchan, nchan), dtype=np.complex64)

    for i in range(0, nchan):
        f, Pxx = welch(
            X[i, :],
            fs=sfreq,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            average=average,
        )

        cs[:, i, i] = Pxx

        for j in range(i + 1, nchan):
            # NOTE: By convention, Pxy is computed with the conjugate FFT
            # of X multiplied by the FFT of Y (scipy documentation)
            # Pxy = <x* y> ~ exp(1j * (phi_y - phi_x))
            _, Pxy = csd(
                X[i, :],
                X[j, :],
                fs=sfreq,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                average=average,
            )

            cs[:, i, j] = Pxy
            cs[:, j, i] = np.conjugate(Pxy)

    return f, cs


def data2cs_fourier(data, sfreq, fres=1, overlap=0.5, normalise=False):
    """
    Parameters
    ----------
    data: array, shape=(channels, samples, [epochs])
        Continuous or epoched data for the calculation of cross-spectral density.
    sfreq: int
        The sampling frequency of the data (in Hz).
    fres: int
        The frequency resolution of the estimated cross-spectral density (step
        size between successive frequency bins, in Hz). Defaults to 1.
    overlap: float
        Overlap of windows used for Welch's method (0 - no overlap, 1 - full overlap).
        Defaults to 0.5 (50% overlap).
    normalize: bool
        Scale the data if True. Defaults to False, might be not necessary in future
    """

    assert len(data.shape) < 3, "Expected continuous data"

    nfft = int(sfreq / fres)
    nperseg = nfft
    noverlap = round(nperseg * overlap)

    # Demean and optionally normalize
    data = data - np.mean(data, axis=(0, 1), keepdims=True)
    if normalise:
        data = data / np.std(data, axis=(0, 1), keepdims=True)

    # Estimate cross-spectra using Welch's method
    f, cs = cpsd_welch(data, sfreq, nfft=nfft, nperseg=nperseg, noverlap=noverlap)

    return f, cs


def cs2coh(cs: np.array):
    nfreq = cs.shape[0]
    coh = cs

    for ifreq in range(0, nfreq):
        pxx = np.sqrt(np.diag(np.abs(cs[ifreq, :, :])))
        csxx = np.squeeze(cs[ifreq, :, :])
        coh[ifreq, :, :] = csxx / (pxx[:, np.newaxis] @ pxx[np.newaxis, :])

    return np.squeeze(coh)


def cohy2con(cohy, measure, return_abs=False):
    if measure == "imcoh":
        result = np.imag(cohy)
    elif measure == "coh":
        result = np.abs(cohy)
    else:
        raise ValueError(f"Measure {measure} is not supported")

    if return_abs:
        return np.abs(result)
    return result


def imcoh(tcs, sfreq, fres=1):
    f, cs = data2cs_fourier(tcs, sfreq, fres=fres)
    cohy = cs2coh(cs)
    imcoh = cohy2con(cohy, measure="imcoh", return_abs=True)

    return f, np.squeeze(imcoh[:, 0, 1])


def spurious_coherence(cohy_spectra):
    """
    Parameters
    ----------
    cohy_spectra: array (freqs,)
        The spectra of coherency.

    Returns
    -------
    spurious_coherence: float
        The value of spurious coherence.
    """

    # Take absolute part to get coherence
    coh_spectra = cohy2con(cohy_spectra, measure="coh")

    # For now, we only use minimal value over all frequencies
    return np.min(coh_spectra)


def estimate_noise_floor(
    sfreq, duration, fstim, conn_params, n_simulations=1000, random_state=None
):
    t_noise = np.arange(sfreq * duration) / sfreq

    conn_noise = np.zeros((n_simulations,))
    seeds = np.random.SeedSequence(random_state).generate_state(n_simulations)
    for i_run, seed in enumerate(tqdm(seeds, desc="Estimating the noise floor")):
        n = white_noise(2, t_noise, random_state=seed)
        f, cs = data2cs_fourier(n, sfreq, fres=1.0, overlap=0.0)
        band = np.logical_and(f > fstim - 1, f < fstim + 1)
        cohy = cs2coh(cs)[band, :, :].mean(axis=0)
        conn_noise[i_run] = cohy2con(cohy, **conn_params)[0, 1]

    return conn_noise


def rift_coherence(
    data, stim, sfreq, fstim, kind, measure, measure_kwargs=dict(), fres=1, overlap=0
):
    assert kind in ["brain_stimulus", "brain_brain"]
    assert measure in ["coh", "imcoh"]

    # Concatenate stimulus to the MEG data, then concatenate epochs
    data_appended = np.concatenate([data, stim[:, np.newaxis, :]], axis=1)
    data_appended = np.concatenate(data_appended, axis=1)

    # Calculate coherence between all pairs of channels
    # NOTE: overlap = 0 is desirable to only use continuous data segments
    f, cs = data2cs_fourier(data_appended, sfreq=sfreq, fres=fres, overlap=overlap)
    cohy = cs2coh(cs)
    coh = cohy2con(cohy, measure=measure, **measure_kwargs)

    # Focus only on the band of interest
    band = np.logical_and(f > fstim - 1, f < fstim + 1)
    assert sum(band) == 1, "Expected only one frequency bin"
    coh_fstim = np.squeeze(coh[band, :, :])

    # Pick brain-stimulus or brain-brain values
    if kind == "brain_stimulus":
        # Pick only coherence between MEG channels / ROIs and the stimulation channel
        coh_fstim = coh_fstim[:-1, -1]
    elif kind == "brain_brain":
        # Pick only coherence between MEG channels / ROIs
        coh_fstim = coh_fstim[:-1, :-1]
    else:
        raise ValueError("unreachable")

    return coh_fstim
