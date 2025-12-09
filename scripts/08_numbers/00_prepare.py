import numpy as np

from ctfeval.config import paths, params
from ctfeval.datasets import get_lemon_subject_ids, get_lemon_filename
from ctfeval.io import load_raw


def get_lemon_durations(subject_ids):
    conditions = ["EO", "EC"]
    num_conditions = len(conditions)
    durations = np.zeros((num_conditions, len(subject_ids)))
    for i_subject, subject_id in enumerate(subject_ids):
        for i_cond, condition in enumerate(conditions):
            raw_path = get_lemon_filename(subject_id, condition, params.lemon_bids)
            raw = load_raw(raw_path)
            durations[i_cond, i_subject] = raw.duration

    return durations


def main():
    output_folder = paths.derivatives / "numbers"
    output_folder.mkdir(exist_ok=True)

    lemon_subject_ids = get_lemon_subject_ids(False)
    # NOTE: two subjects were skipped in grand averages because of a mismatch
    # in sampling frequency (100 Hz instead of 250 Hz)
    lemon_subject_ids.remove("sub-010276")
    lemon_subject_ids.remove("sub-010277")
    assert len(lemon_subject_ids) == 200

    durations = get_lemon_durations(lemon_subject_ids)
    np.save(output_folder / "durations.npy", durations)


if __name__ == "__main__":
    main()
