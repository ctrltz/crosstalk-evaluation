import numpy as np

from meegsim.waveform import white_noise

from ctfeval.metrics import estimate_correlation
from ctfeval.prepare import prepare_filter


def test_estimate_correlation():
    sfreq = 250
    duration = 60
    times = np.arange(sfreq * duration) / sfreq
    s1 = white_noise(1, times, random_state=1)
    s2 = white_noise(1, times, random_state=2)

    b, a = prepare_filter(sfreq, 8, 12)
    assert np.isclose(estimate_correlation(s1, s1, b, a), 1), "copy"
    assert estimate_correlation(s1, s2, b, a) < 0.1, "random"
