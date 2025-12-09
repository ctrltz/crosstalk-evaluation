import mne
import subprocess

from ctfeval.config import paths
from ctfeval.log import logger


SCHAEFER_BASE_URL = "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label/{}"


def download_fsaverage():
    if not paths.subjects_dir.exists():
        raise ValueError("Provided subjects_dir does not exist. Check the .env file")

    logger.info("Fetching fsaverage data")
    mne.datasets.fetch_fsaverage(subjects_dir=paths.subjects_dir)
    logger.info("Done")


def download_schaefer_atlas(n_parcels=400, n_networks=17):
    assert n_parcels in [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    ], "Bad number of parcels"
    assert n_networks in [7, 17], "Bad number of networks"

    logger.info("Checking if the Schaefer atlas is installed")
    label_path = paths.subjects_dir / "fsaverage" / "label"
    for hemi in ["lh", "rh"]:
        fname = (
            f"{hemi}.Schaefer2018_{n_parcels}Parcels_{n_networks}Networks_order.annot"
        )
        hemi_path = label_path / fname
        if hemi_path.exists():
            logger.info(f"{fname} exists")
            continue

        logger.info(f"Downloading {fname}")
        subprocess.run(
            ["wget", "-4", "-O", label_path / fname, SCHAEFER_BASE_URL.format(fname)]
        )
    logger.info("Done")


def download_hcp_parcellation():
    if not paths.subjects_dir.exists():
        raise ValueError("Provided subjects_dir does not exist. Check the .env file")

    logger.info("Fetching HCP parcellation")
    mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=paths.subjects_dir)
    logger.info("Done")


def main():
    # Create a file to use as a dependency in Snakemake rules
    with open(".workspace", "w+") as f:
        f.write("init")

    # Set up the project structure
    folders_to_create = [
        paths.data,
        paths.derivatives,
        paths.examples,
        paths.figures,
        paths.figure_components,
        paths.numbers,
        paths.precomputed,
        paths.results,
        paths.review,
        paths.rift,
        paths.rift_results,
        paths.sanity,
        paths.simulations,
        paths.theory,
        paths.toml,
    ]
    for folder in folders_to_create:
        logger.info(f"Creating {folder}")
        folder.mkdir(exist_ok=True, parents=True)

    # Download required data and create a Snakemake checkpoint
    download_fsaverage()
    download_schaefer_atlas()
    download_hcp_parcellation()
    with open(paths.subjects_dir / ".ready", "w+") as f:
        f.write("init")


if __name__ == "__main__":
    main()
