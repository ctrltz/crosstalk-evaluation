import argparse
import numpy as np
import pandas as pd

from itertools import product

from ctfeval.config import paths, params
from ctfeval.evaluate import get_correlation_across_rois
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.utils import get_simulation_names_for_experiment


parser = argparse.ArgumentParser()
parser.add_argument("--parcellation", choices=PARCELLATIONS.keys(), default="DK")
parser.add_argument("--experiment", choices=["1", "2", "3", "4a", "4b"], required=True)


def collect_results(results_path, p, experiment):
    simulations = get_simulation_names_for_experiment(
        p.code, params.spacing_sim, experiment
    )
    methods = [
        f"{p.code}_{spacing}_baseline"
        for spacing in [params.spacing_sim, params.spacing_rec]
    ]

    rat_corr_data = []
    rec_data = []

    for simulation, method in list(product(simulations, methods)):
        spacing_rec = method.split("_")[1]
        spacing_desc = "same" if spacing_rec == "oct6" else "another"

        # Load the results
        # NOTE: with correlation, we need to take the absolute value to
        # account for potential sign flips
        combined_path = results_path / simulation / method / "combined"
        rec_file = combined_path / "results.npy"
        rec = np.abs(np.load(rec_file))
        rec_mean = rec.mean(axis=2)

        pipe_file = combined_path / "pipelines.npy"
        pipelines = np.load(pipe_file)

        # Pack the results into a dataframe
        for i_pipe, pipeline in enumerate(pipelines):
            for i_label, label in enumerate(p.labels):
                rec_data.append(
                    {
                        "simulation": simulation,
                        "method": method,
                        "pipeline": " | ".join(pipeline),
                        "inv_method": pipeline[0],
                        "reg": pipeline[1],
                        "roi_method": pipeline[2],
                        "label": label.name,
                        "rec_mean": rec_mean[i_pipe, i_label],
                    }
                )

        # NOTE: theory_same_grid also contains all options from theory_other_grid,
        # so in combination with the continue block this loop covers both parts
        for theory_mode in params.theory_same_grid:
            crosstalk_file = f"crosstalk_{theory_mode}.npy"
            crosstalk_path = combined_path / crosstalk_file
            if not crosstalk_path.exists():
                logger.info(f"Skipping {crosstalk_path}")
                continue

            logger.info(f"Processing {crosstalk_path}")
            chunks = theory_mode.split("_")
            mask_mode = chunks[0]
            data_cov_mode = chunks[-1]
            source_cov_mode = f"{chunks[1]}_{chunks[2]}"

            ratio = np.sqrt(np.load(crosstalk_path))
            ratio_mean = ratio.mean(axis=2)
            r_rat, beta_rat = get_correlation_across_rois(ratio_mean, rec_mean)

            eval_desc = "dCTF" if data_cov_mode == "actual" else "CTF"
            for i_pipeline, pipeline in enumerate(pipelines):
                rat_info = {
                    "simulation": simulation,
                    "method": method,
                    "pipeline": " | ".join(pipeline),
                    "inv_method": pipeline[0],
                    "reg": pipeline[1],
                    "roi_method": pipeline[2],
                    "metric": f"{eval_desc} | {spacing_desc} | {mask_mode} | {source_cov_mode}",
                    "mask_mode": mask_mode,
                    "source_cov_mode": source_cov_mode,
                    "data_cov_mode": data_cov_mode,
                    "corr": r_rat[i_pipeline],
                    "beta": beta_rat[i_pipeline],
                }
                rat_corr_data.append(rat_info)

    rec_df = pd.DataFrame(rec_data)
    rat_corr_df = pd.DataFrame(rat_corr_data)

    return rec_df, rat_corr_df


def main(parcellation, experiment):
    p = PARCELLATIONS[parcellation].load("fsaverage", paths.subjects_dir)
    results_path = paths.derivatives / "simulations"
    save_path = results_path / "experiments"
    save_path.mkdir(exist_ok=True)

    rec_df, rat_corr_df = collect_results(results_path, p, experiment)
    rec_df.to_csv(save_path / f"exp{experiment}_{parcellation}_rec.csv", index=False)
    rat_corr_df.to_csv(
        save_path / f"exp{experiment}_{parcellation}_corr_rat.csv", index=False
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(parcellation=args.parcellation, experiment=args.experiment)
