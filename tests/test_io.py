import mne
import numpy as np

from mne.io.constants import FIFF

from ctfeval.config import paths, params
from ctfeval.io import (
    load_head_model,
    dict_to_npz,
    dict_from_npz,
    init_subject,
    load_simulation,
)

from utils.prepare import dummy_simulation


def test_load_head_model():
    lemon_info = mne.io.read_info(paths.lemon_info)
    fwd = load_head_model("fsaverage", paths.subjects_dir, info=lemon_info)

    assert fwd["source_ori"] == FIFF.FIFFV_MNE_FIXED_ORI, "Expected fixed orientations"
    assert (
        fwd.ch_names == lemon_info.ch_names
    ), "Expected the channel order to match info"


def test_dict_to_npz_no_transform():
    prepared = dict_to_npz(a={"v1": 1, "v2": "test"}, b={"v3": 20})
    assert len(prepared) == 3
    assert prepared.get("a__v1", "") == 1
    assert prepared.get("a__v2", "") == "test"
    assert prepared.get("b__v3", "") == 20


def test_dict_to_npz_with_transform():
    prepared = dict_to_npz(transform=lambda x: 10 * x, a={"v1": 1, "v2": 2})
    assert len(prepared) == 2
    assert prepared.get("a__v1", "") == 10
    assert prepared.get("a__v2", "") == 20


def test_dict_from_npz_no_transform():
    original = {"a__v1": 1, "b__v2": 2, "a_v1": "trap"}
    extracted = dict_from_npz(original, "a")

    assert len(extracted) == 1
    assert extracted.get("v1", "") == 1


def test_dict_from_npz_with_transform():
    original = {
        "a__v1": 1,
        "a__v2": 2,
    }
    extracted = dict_from_npz(original, "a", transform=lambda x: 10 * x)

    assert len(extracted) == 2
    assert extracted.get("v1", "") == 10
    assert extracted.get("v2", "") == 20


def test_load_save_simulation():
    """
    A combined test for loading/saving a simulation.
    """
    simulation_path, simulation_id, sc = dummy_simulation(return_sc=True)
    fwd, _, lemon_info, p = init_subject(parcellation="DK")
    raw = sc.to_raw(fwd, lemon_info, params.sensor_noise_level)

    raw_loaded, gt_loaded, names_loaded = load_simulation(
        simulation_path, simulation_id
    )
    assert list(names_loaded) == p.label_names
    for i, name in enumerate(p.label_names):
        assert np.allclose(gt_loaded[i, :], sc[name].waveform)
    assert np.allclose(raw.get_data(), raw_loaded.get_data())
