import argparse
import matplotlib.pyplot as plt
import mne

from ctfeval.config import paths
from ctfeval.prepare import prepare_source_space, prepare_forward


parser = argparse.ArgumentParser()
parser.add_argument(
    "--spacing",
    choices=["oct6", "ico4"],
    required=True,
    help="Spacing of the source space to use",
)


def main(spacing, subject="fsaverage", make_plots=False):
    plot_path = paths.results / "prepare"
    plot_path.mkdir(parents=True, exist_ok=True)

    output_folder = plot_path / spacing
    output_folder.mkdir(exist_ok=True)

    # Source space
    src = prepare_source_space(
        paths.subjects_dir,
        subject,
        subfolder=paths.head_model_subdir,
        spacing=spacing,
        plot_src=False,
        save=True,
        overwrite=True,
    )

    # Load an exemplary LEMON recording with all channels
    lemon_info = mne.io.read_info(paths.lemon_info)
    if make_plots:
        alignment = mne.viz.plot_alignment(
            lemon_info,
            src=src,
            eeg=["original", "projected"],
            trans="fsaverage",
            show_axes=True,
            mri_fiducials=True,
            dig="fiducials",
            subject=subject,
            subjects_dir=paths.subjects_dir,
        )
        screenshot = alignment.plotter.screenshot()
        alignment.plotter.close()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(screenshot)
        ax.axis("off")
        fig.savefig(output_folder / "alignment.png")
        plt.close(fig)

    # Forward model
    prepare_forward(
        paths.subjects_dir,
        subject,
        src,
        lemon_info,
        subfolder=paths.head_model_subdir,
        spacing=spacing,
        save=True,
        overwrite=True,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.spacing)
