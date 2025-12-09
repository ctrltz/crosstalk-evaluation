import numpy as np
import pytest

from meegsim.waveform import white_noise

from ctfeval.connectivity import (
    data2cs_fourier,
    data2cs_hilbert,
    cs2coh,
    cohy2con,
    rift_coherence,
)


@pytest.mark.parametrize(
    "fres,dphi",
    [
        [1, 0],
        [1, np.pi / 4],
        [1, np.pi / 2],
        [1, np.pi],
        [0.5, np.pi / 2],
        [2, np.pi / 2],
    ],
)
def test_data2cs_fourier_sine(fres, dphi):
    # Generate two 10 Hz sinusoids with a known delay
    sfreq = 250
    duration = 60
    fpeak = 10
    dphi = np.pi / 2

    t = np.arange(sfreq * duration) / sfreq
    y1 = np.cos(2 * np.pi * fpeak * t)
    y2 = np.cos(2 * np.pi * fpeak * t + dphi)
    data = np.vstack((y1, y2))

    expected_cs = np.array(
        [[1.0 + 0.0j, np.exp(1.0j * dphi)], [np.exp(-1.0j * dphi), 1.0 + 0.0j]]
    )

    f, cs = data2cs_fourier(data, sfreq, fres=fres)

    # Check the frequency resolution
    assert np.all(np.diff(f) == fres)

    # Check that the estimated cross-spectra is correct up to a constant
    # multiplier (checked via the cosine similarity of flattened arrays)
    actual_cs = np.ravel(np.squeeze(cs[f == fpeak, :, :]))
    expected_cs = np.ravel(expected_cs)

    for part_fun in [np.real, np.imag]:
        actual_cs_part = part_fun(actual_cs)
        actual_cs_part /= np.linalg.norm(actual_cs_part)

        expected_cs_part = part_fun(expected_cs)
        expected_cs_part /= np.linalg.norm(expected_cs_part)

        assert np.allclose(np.dot(actual_cs_part, expected_cs_part), 1.0)


@pytest.mark.parametrize(
    "dphi",
    [
        0,
        np.pi / 4,
        np.pi / 2,
        np.pi,
    ],
)
def test_data2cs_hilbert_sine(dphi):
    # Generate two 10 Hz sinusoids with a known delay
    sfreq = 250
    duration = 60
    fpeak = 10
    dphi = np.pi / 2

    t = np.arange(sfreq * duration) / sfreq
    y1 = np.cos(2 * np.pi * fpeak * t)
    y2 = np.cos(2 * np.pi * fpeak * t + dphi)
    data = np.vstack((y1, y2))

    expected_cs = np.array(
        [[1.0 + 0.0j, np.exp(1.0j * dphi)], [np.exp(-1.0j * dphi), 1.0 + 0.0j]]
    )

    cs = data2cs_hilbert(data)

    # Check that the estimated cross-spectra is correct up to a constant
    # multiplier (checked via the cosine similarity of flattened arrays)
    actual_cs = np.ravel(cs)
    expected_cs = np.ravel(expected_cs)

    for part_fun in [np.real, np.imag]:
        actual_cs_part = part_fun(actual_cs)
        actual_cs_part /= np.linalg.norm(actual_cs_part)

        expected_cs_part = part_fun(expected_cs)
        expected_cs_part /= np.linalg.norm(expected_cs_part)

        assert np.allclose(np.dot(actual_cs_part, expected_cs_part), 1.0)


def test_cs2coh():
    # A simple test of coherence calculations with different phase lags
    cs = np.array(
        [
            [1.0 + 0.0j, 0.0 + 2.0j, -3.0 + 0.0j],
            [0.0 - 2.0j, 4.0 + 0.0j, 3 * np.sqrt(2) / 2 + 3.0j * np.sqrt(2) / 2],
            [-3 + 0.0j, 3 * np.sqrt(2) / 2 - 3.0j * np.sqrt(2) / 2, 9.0 + 0.0j],
        ]
    )[np.newaxis, :, :]

    expected_coh = np.array(
        [
            [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j],
            [0.0 - 1.0j, 1.0 + 0.0j, np.sqrt(2) / 4 + np.sqrt(2) * 1.0j / 4],
            [-1 + 0.0j, np.sqrt(2) / 4 - np.sqrt(2) * 1.0j / 4, 1.0 + 0.0j],
        ]
    )

    coh = cs2coh(cs)

    assert np.array_equal(coh, expected_coh)


@pytest.mark.parametrize(
    "measure,return_abs,expected",
    [
        (
            "imcoh",
            False,
            np.array(
                [
                    0,
                    1,
                    0,
                    -1,
                    1 / np.sqrt(2),
                    1 / np.sqrt(2),
                    -1 / np.sqrt(2),
                    -1 / np.sqrt(2),
                    -0.25,
                    np.sqrt(3) / 4,
                ]
            ),
        ),
        (
            "imcoh",
            True,
            np.array(
                [
                    0,
                    1,
                    0,
                    1,
                    1 / np.sqrt(2),
                    1 / np.sqrt(2),
                    1 / np.sqrt(2),
                    1 / np.sqrt(2),
                    0.25,
                    np.sqrt(3) / 4,
                ]
            ),
        ),
        ("coh", True, np.array([1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5])),
    ],
)
def test_cohy2con(measure, return_abs, expected):
    cohy_test = np.array(
        [
            1 + 0j,
            0 + 1j,
            -1 + 0j,
            0 - 1j,
            (1 + 1j) / np.sqrt(2),
            (-1 + 1j) / np.sqrt(2),
            (-1 - 1j) / np.sqrt(2),
            (1 - 1j) / np.sqrt(2),
            (np.sqrt(3) - 1j) / 4,
            (-1 + np.sqrt(3) * 1j) / 4,
        ]
    )

    con = cohy2con(cohy_test, measure, return_abs)
    assert np.array_equal(con, expected)


def test_rift_coherence():
    """
    Only testing the shape of the output depending on the `kind` value.
    """
    n_channels = 10
    n_samples = 600
    n_epochs = 10
    sfreq = 600
    fstim = 60

    times = np.arange(n_samples) / sfreq
    brain_data = np.stack(
        [white_noise(n_channels, times) for _ in range(n_epochs)], axis=0
    )
    stim_data = np.squeeze(
        np.stack([white_noise(1, times) for _ in range(n_epochs)], axis=0)
    )

    brain_stim_coh = rift_coherence(
        brain_data, stim_data, sfreq, fstim, kind="brain_stimulus", measure="coh"
    )
    assert brain_stim_coh.shape == (n_channels,)

    brain_brain_imcoh = rift_coherence(
        brain_data, stim_data, sfreq, fstim, kind="brain_brain", measure="imcoh"
    )
    assert brain_brain_imcoh.shape == (n_channels, n_channels)
