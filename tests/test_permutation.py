import numpy as np

from ctfeval.permutation import permute


def test_permute_shuffles_samples():
    num_channels = 10
    num_samples = 10000
    sfreq = 250
    num_blocks = num_samples // sfreq

    # Data is the same for all blocks - no effect of permutations as long as
    # window is a multiple of sfreq
    original = np.tile(np.arange(sfreq)[np.newaxis, :], (num_channels, num_blocks))
    permuted = permute(original, sfreq, seg_len=2, random_state=123)
    assert np.allclose(original, permuted)

    permuted = permute(original, sfreq, seg_len=1, random_state=123)
    assert np.allclose(original, permuted)

    # Data should change for other window sizes
    original = np.tile(np.arange(sfreq)[np.newaxis, :], (num_channels, num_blocks))
    permuted = permute(original, sfreq, seg_len=1.5, random_state=123)
    assert not np.allclose(original, permuted)


def test_permute_keeps_channels():
    num_channels = 10
    num_samples = 10000
    sfreq = 250

    # Channel data is constant so permutations should not have any effect
    original = np.tile(np.arange(num_channels)[:, np.newaxis], (1, num_samples))
    permuted = permute(original, sfreq, seg_len=2, random_state=123)
    assert np.allclose(original, permuted)
