import mne

from ctfeval.config import paths


def test_lemon_info():
    """
    Make sure that LEMON info matches expectations.
    """
    lemon_info = mne.io.read_info(paths.lemon_info)
    assert len(lemon_info.ch_names) == 61, "Number of channels does not match"
    assert len(lemon_info["projs"]) == 1, "Expected 1 projector"

    proj = lemon_info["projs"][0]
    assert (
        "Average EEG reference" in proj["desc"]
    ), "Expected average reference projector"
