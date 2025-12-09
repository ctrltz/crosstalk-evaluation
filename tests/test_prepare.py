import mne
import numpy as np
import pytest

from ctfeval.config import paths
from ctfeval.io import load_head_model
from ctfeval.prepare import (
    prepare_source_space,
    setup_inverse_operator,
    interpolate_missing_channels,
)


@pytest.mark.parametrize(
    "spacing,expected_num_vertices", [("oct6", 4098 * 2), ("ico4", 2562 * 2)]
)
def test_prepare_source_space(spacing, expected_num_vertices):
    src = prepare_source_space(
        paths.subjects_dir,
        "fsaverage",
        subfolder=paths.head_model_subdir,
        spacing=spacing,
    )
    num_vertices = sum(s["nuse"] for s in src)
    assert num_vertices == expected_num_vertices, spacing


def test_setup_inverse_operator():
    lemon_info = mne.io.read_info(paths.lemon_info)
    fwd = load_head_model("fsaverage", paths.subjects_dir, info=lemon_info)
    inv = setup_inverse_operator(fwd, lemon_info)

    assert (
        inv.ch_names == lemon_info.ch_names
    ), "Expected the channel order to match info"
    assert np.allclose(
        inv["noise_cov"].data, np.eye(fwd["nchan"])
    ), "Expected an identity noise covariance matrix"


def test_interpolate_missing_channels():
    lemon_info = mne.io.read_info(paths.lemon_info)

    # Dummy data with some channels present
    sfreq = 250
    ch_names = lemon_info.ch_names[::2]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((len(ch_names), sfreq * 4)), info)

    raw_interp = interpolate_missing_channels(raw, lemon_info)
    assert raw_interp.ch_names == lemon_info.ch_names
