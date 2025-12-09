import mne
import numpy as np
import pandas as pd

from tqdm import tqdm

from ctfeval.config import paths, params
from ctfeval.ctf import get_max_ctf_ratios
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.prepare import prepare_forward, prepare_source_space


def max_ratios_vs_num_channels(p, src, num_channels):
    max_ratios = np.zeros((p.n_labels, len(num_channels)))
    for i_setup, ch_count in enumerate(tqdm(num_channels)):
        # Create an mne.Info object using all sensors from the Biosemi montage
        logger.info(f"Creating Info with {ch_count} sensors")
        montage_name = f"biosemi{ch_count}"
        montage = mne.channels.make_standard_montage(montage_name)
        info = mne.create_info(montage.ch_names, sfreq=250, ch_types="eeg")
        info.set_montage(montage_name)

        fwd = prepare_forward(
            paths.subjects_dir,
            "fsaverage",
            src,
            info,
            meg=False,
            eeg=True,
            mindist=5.0,
            subfolder="bem",
            spacing="oct6",
            bem_file="fsaverage-5120-5120-5120-bem-sol.fif",
            trans="fsaverage",
            save=False,
            overwrite=False,
        )
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
        max_ratios[:, i_setup] = get_max_ctf_ratios(fwd, p)
        logger.info("Obtained max CTF ratios")

    return max_ratios


def main():
    src = prepare_source_space(
        paths.subjects_dir, "fsaverage", spacing="oct6", plot_src=False
    )
    save_path = paths.theory / "parcellations"

    for code in ["DK"]:
        p = PARCELLATIONS[code].load("fsaverage", paths.subjects_dir)
        max_ratios = max_ratios_vs_num_channels(p, src, params.max_ratio_num_channels)

        ratio_data = []
        for i_setup, ch_count in enumerate(params.max_ratio_num_channels):
            for i_label, label in enumerate(p.labels):
                ratio_data.append(
                    {
                        "ch_count": str(ch_count),
                        "label": label.name,
                        "max_ratio": max_ratios[i_label, i_setup],
                    }
                )

        df_ratio = pd.DataFrame(ratio_data)
        df_ratio.to_csv(save_path / f"{code}_num_channels.csv", index=False)


if __name__ == "__main__":
    main()
