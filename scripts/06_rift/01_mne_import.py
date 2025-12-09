import argparse
import mne
import numpy as np

from mne.io.fieldtrip.utils import _validate_ft_struct
from mne.utils.check import _import_pymatreader_funcs

from ctfeval.config import paths
from ctfeval.datasets import get_rift_subject_ids, rift_create_info, rift_subfolder
from ctfeval.log import logger


parser = argparse.ArgumentParser()
parser.add_argument("--cond", choices=["onestim", "twostim"], required=True)
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)


def import_subject(preproc_folder, info_folder, subject_id, cond):
    logger.info(f"Processing {subject_id}: {cond}")

    # Raw info
    logger.info("    Looking for raw info")
    numeric_id = subject_id.partition('sub')[-1]
    subject_meg_path = paths.rift_raw / f"sub-{numeric_id}" / "ses-meg01" / "meg"
    folder_path = list(subject_meg_path.glob(f"{subject_id}*"))
    assert (
        len(folder_path) == 1 and folder_path[0].is_dir()
    ), "Expected only one matching folder with raw data"
    raw = mne.io.read_raw_ctf(str(folder_path[0]), clean_names=True)
    raw_info = raw.info
    logger.info("    Done")

    # Prepare FieldTrip structure
    logger.info("    Reading FieldTrip structure")
    read_mat = _import_pymatreader_funcs("FieldTrip I/O")
    dataset_name = "data_" + cond
    file_path = preproc_folder / f"{subject_id}-data_{cond}.mat"
    ft_struct = read_mat(
        file_path, ignore_fields=["previous"], variable_names=[dataset_name]
    )
    ft_struct = ft_struct[dataset_name]
    _validate_ft_struct(ft_struct)
    trials = ft_struct["trial"]
    trials = np.array(trials)
    logger.info("    Done")

    # Save stim channels before dropping them
    logger.info("    Saving stimulation data")
    for ch_label in ["UADC001", "tag1", "tag2"]:
        ch_idx = np.where(np.array(ft_struct["label"]) == ch_label)[0][0]
        ch_name = "stimulation" if ch_label == "UADC001" else ch_label
        np.save(
            preproc_folder / f"{subject_id}_data_{cond}_{ch_name}.npy",
            trials[:, ch_idx, :],
        )
    logger.info("    Done")

    # Create info
    logger.info("    Creating and adjusting mne.Info")
    info = rift_create_info(ft_struct, raw_info)  # create info structure
    assert ft_struct["label"] == info.ch_names
    logger.info("    Done")

    # save epochs
    logger.info("    Saving epochs")
    trials = np.array(ft_struct["trial"])
    epochs = mne.EpochsArray(trials, info, tmin=ft_struct["time"][0][0])
    epochs.save(preproc_folder / f"{subject_id}_data_{cond}-epo.fif", overwrite=True)
    logger.info("    Done")

    # save info
    logger.info("    Saving Info")
    epochs.info.save(info_folder / f"{subject_id}_{cond}-info.fif")
    logger.info("    Done")

    logger.info(f"Done with {subject_id}: {cond}")


def main(cond, tagging_type, random_phases):
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    preproc_folder = tag_folder / "preproc"
    info_folder = tag_folder / "info"
    info_folder.mkdir(exist_ok=True)

    logger.info(f"Tag folder: {tag_folder}")
    logger.info(f"Preproc folder: {preproc_folder}")
    logger.info(f"Info folder: {info_folder}")

    for subject_id in get_rift_subject_ids():
        import_subject(preproc_folder, info_folder, subject_id, cond)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.cond, int(args.tagging_type), int(args.random_phases))
