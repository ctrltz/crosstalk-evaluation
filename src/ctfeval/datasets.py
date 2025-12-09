from mne import pick_info
from mne.io.fieldtrip.utils import _set_sfreq, _remove_missing_channels_from_trial

from ctfeval.config import paths


def get_lemon_subject_ids(assume_bids):
    if assume_bids:
        # If BIDS data is used, each subject has a separate folder, return their names
        subject_ids = [p.name for p in paths.lemon_data.glob("sub-*") if p.is_dir()]
    else:
        # Otherwise, all files are mixed for EC and EO, get unique IDs
        subject_files = [p.name for p in paths.lemon_data.glob("sub-*.set")]
        subject_ids = list(set([filename[:-7] for filename in subject_files]))

    # Keep only participants for whom raw data is also available
    raw_dirs = [p.name for p in paths.lemon_raw_data.iterdir() if p.is_dir()]
    return [sid for sid in subject_ids if sid in raw_dirs]


def get_lemon_age(age_str):
    lo, hi = [int(x) for x in age_str.split("-")]
    return 0.5 * (lo + hi)


def get_lemon_filename(subject, condition, assume_bids):
    if assume_bids:
        return paths.lemon_data / subject / f"{subject}_{condition}.set"

    return paths.lemon_data / f"{subject}_{condition}.set"


def get_rift_subject_ids(strip_sub=False):
    # NOTE: all subjects that were included in the original study have
    # precomputed SNR and spectra, we use the same IDs
    subject_ids = sorted(
        [
            p.name.partition("-snr")[0]
            for p in paths.rift_scratch.glob("sub*-snr-and-spectra.mat")
        ]
    )

    if not strip_sub:
        return subject_ids

    return [int(sid.removeprefix("sub")) for sid in subject_ids]


def rift_subfolder(tagging_type, random_phases):
    return f"tag_type_{tagging_type}_random_phases_{random_phases}"


def rift_create_info(ft_struct, raw_info):
    """
    Create MNE info structure from a FieldTrip structure.
    Adapted from the function mne.io.fieldtrip.utils._create_info
    """
    sfreq = _set_sfreq(ft_struct)
    ch_names = ft_struct["label"]
    info = raw_info.copy()

    missing_channels = set(ch_names) - set(info["ch_names"])
    missing_chan_idx = [ch_names.index(ch) for ch in missing_channels]
    new_chs = [ch for ch in ch_names if ch not in missing_channels]
    ch_names = new_chs
    ft_struct["label"] = ch_names

    if "trial" in ft_struct:
        ft_struct["trial"] = _remove_missing_channels_from_trial(
            ft_struct["trial"], missing_chan_idx
        )

    with info._unlock():
        info["sfreq"] = sfreq
    ch_idx = [info["ch_names"].index(ch) for ch in ch_names]
    pick_info(info, ch_idx, copy=False)
    assert ft_struct["label"] == info.ch_names
    return info
