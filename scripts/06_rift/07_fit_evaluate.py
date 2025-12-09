import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm

from ctfeval.config import paths, params
from ctfeval.datasets import rift_subfolder
from ctfeval.io import load_source_space
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.rift.fit_evaluate import define_search_grid, fit_evaluate
from ctfeval.rift.metrics import METRICS


parser = argparse.ArgumentParser()
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)
parser.add_argument("--model", required=True, choices=["no_leakage", "distance", "ctf"])
parser.add_argument("--kind", required=True, choices=["brain_stimulus", "brain_brain"])


def main(tagging_type, random_phases, model, kind):
    # Paths
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    theory_path = tag_folder / "theory"
    results_path = tag_folder / kind
    ctf_file = paths.rift / "ctf_array.npy"
    csv_file = results_path / f"search_results_{model}.csv"

    # Load the noise floor estimate
    onestim = kind == "brain_stimulus"
    suffix = "coh_onestim" if onestim else "imcoh_twostim"
    with np.load(theory_path / f"noise_floor_{suffix}.npz") as data:
        coh_threshold = data["conn_threshold"]
    logger.info(f"{coh_threshold=}")

    # Load the head model and filters
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    ctf_array = np.load(ctf_file)

    # Define the search space
    grid = define_search_grid(
        src,
        p[params.rift_lh_search_space],
        p[params.rift_rh_search_space],
        params.rift_gammas,
        model,
        hemi="both",
    )

    # Load real data, preserve the order of methods
    filename = "onestim_avg_abscoh_stim.npy" if onestim else "twostim_avg_absimcoh.npy"
    coh_real = np.load(results_path / filename)
    logger.info(f"Loaded real data results, {coh_real.shape=}")

    # Run the grid search
    results = []
    for lh_vertno, rh_vertno, gamma in tqdm(grid, desc="Grid search"):
        result = fit_evaluate(
            ctf_array,
            p.labels,
            model,
            kind,
            coh_real,
            coh_threshold,
            src,
            lh_vertno,
            rh_vertno,
            gamma,
            METRICS,
        )
        results.append(result)

    # Save the results
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        tagging_type=int(args.tagging_type),
        random_phases=int(args.random_phases),
        model=args.model,
        kind=args.kind,
    )
