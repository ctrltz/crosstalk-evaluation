import argparse
import json
import numpy as np

from tqdm import tqdm

from ctfeval.config import paths
from ctfeval.evaluate import evaluate_theory
from ctfeval.io import init_subject
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS


parser = argparse.ArgumentParser()
parser.add_argument("--spacing", choices=["oct6", "ico4"], default="oct6")
parser.add_argument("--parcellation", choices=PARCELLATIONS.keys(), default="DK")
parser.add_argument("--simulation", type=str, required=True)
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--mask", type=str, choices=["roi", "source"], required=True)
parser.add_argument(
    "--source-cov", type=str, choices=["no_info", "full_info"], required=True
)
parser.add_argument("--data-cov", type=str, choices=["auto", "actual"], required=True)


COMBINATIONS = [
    ("roi", "no_info", "auto"),
    ("roi", "no_info", "actual"),
    ("roi", "full_info", "auto"),
    ("roi", "full_info", "actual"),
    ("source", "full_info", "auto"),
    ("source", "full_info", "actual"),
]


def main(
    spacing,
    parcellation,
    simulation_name,
    method_name,
    mask_mode,
    source_cov_mode,
    data_cov_mode,
):
    assert (mask_mode, source_cov_mode, data_cov_mode) in COMBINATIONS
    cfg = dict(
        spacing=spacing,
        parcellation=parcellation,
        simulation_name=simulation_name,
        method_name=method_name,
        mask_mode=mask_mode,
        source_cov_mode=source_cov_mode,
        data_cov_mode=data_cov_mode,
    )
    logger.info(f"Provided parameters: {json.dumps(cfg, indent=4)}")

    # Prepare the head model
    fwd, _, lemon_info, p = init_subject(spacing=spacing, parcellation=parcellation)

    # Set up the paths
    simulation_path = paths.simulations / simulation_name
    save_path = (
        paths.derivatives / "simulations" / simulation_name / method_name / "combined"
    )
    save_path.mkdir(exist_ok=True)

    # Load filters and pipelines
    filters = np.load(save_path / "filters.npy")
    pipelines = np.load(save_path / "pipelines.npy")

    # Load simulations
    with open(simulation_path / "simulations.json") as f:
        simulations_desc = json.load(f)
    simulation_ids = simulations_desc["filenames"]
    n_simulations = len(simulation_ids)
    n_pipelines = len(pipelines)

    ratio = np.zeros((n_pipelines, p.n_labels, n_simulations))
    for i_sim in tqdm(range(n_simulations)):
        simulation_id = simulation_ids[i_sim]
        ratio[:, :, i_sim] = evaluate_theory(
            filters[:, :, i_sim, :],
            pipelines,
            p,
            fwd,
            mask_mode,
            source_cov_mode,
            data_cov_mode,
            simulation_path,
            simulation_id,
        )

    filename = f"crosstalk_{mask_mode}_{source_cov_mode}_{data_cov_mode}.npy"
    np.save(save_path / filename, ratio)
    logger.info(f"Saved the estimated ratio values, shape: {ratio.shape}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        spacing=args.spacing,
        parcellation=args.parcellation,
        simulation_name=args.simulation,
        method_name=args.method,
        mask_mode=args.mask,
        source_cov_mode=args.source_cov,
        data_cov_mode=args.data_cov,
    )
