import argparse
import json
import numpy as np

from tqdm import tqdm

from ctfeval.config import paths, params, get_pipelines
from ctfeval.evaluate import evaluate_sim
from ctfeval.io import load_simulation, init_subject
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.prepare import prepare_filter


parser = argparse.ArgumentParser()
parser.add_argument("--spacing", choices=["oct6", "ico4"], default="oct6")
parser.add_argument("--parcellation", choices=PARCELLATIONS.keys(), default="DK")
parser.add_argument("--label", type=str, default="")
parser.add_argument("--simulation", type=str, required=True)
parser.add_argument("--debug", action="store_true")


def evaluate(
    fwd,
    inv,
    label,
    pipelines,
    metrics,
    simulations,
    simulation_path,
    derivative_path,
    fmin,
    fmax,
    sfreq,
    debug=False,
):
    n_pipelines = len(pipelines)
    n_simulations = len(simulations)
    n_channels = fwd["nchan"]
    results = {m: np.full((n_pipelines, n_simulations), np.nan) for m in metrics}
    filters = np.zeros((n_pipelines, n_simulations, n_channels))

    b, a = prepare_filter(sfreq, fmin, fmax)

    simulations_to_process = simulations
    if debug:
        simulations_to_process = simulations[:5]

    for i_sim, simulation_name in enumerate(
        tqdm(simulations_to_process, desc=label.name)
    ):
        raw, gt, label_names = load_simulation(simulation_path, simulation_name)
        if label.name not in label_names:
            raise RuntimeError(f"Could not find ground truth for label {label.name}")

        label_idx = np.where(label_names == label.name)[0][0]
        label_gt = gt[label_idx, :].copy()

        results_sim, filters[:, i_sim, :] = evaluate_sim(
            fwd, inv, raw, label_gt, label, pipelines, metrics, b, a
        )
        for m in metrics:
            results[m][:, i_sim] = results_sim[m]

    output_path = derivative_path / f"results_{label.name}.npz"

    np.savez(output_path, pipelines=pipelines, filters=filters, **results)


def main(parcellation, simulation_name, label_name=None, debug=False, spacing="oct6"):
    logger.info(f"{spacing=}")
    logger.info(f"{parcellation=}")
    logger.info(f"{simulation_name=}")
    logger.info(f"{label_name=}")

    fwd, inv, _, p = init_subject(spacing=spacing, parcellation=parcellation)
    labels_to_process = p.labels
    if label_name:
        labels_to_process = [p[label_name]]
    logger.info(f"Label(s) to process: {[label.name for label in labels_to_process]}")

    # Simulation
    simulation_path = paths.simulations / simulation_name
    logger.info(f"Simulation path: {simulation_path}")

    # Evaluate
    with open(simulation_path / "simulations.json") as f:
        simulations_desc = json.load(f)
    simulation_ids = simulations_desc["filenames"]
    logger.info(f"Found {len(simulation_ids)} simulations")

    baseline_folder = f"{parcellation}_{spacing}_baseline"
    derivative_path = (
        paths.derivatives / "simulations" / simulation_name / baseline_folder
    )
    derivative_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Derivative path: {derivative_path}")

    # Pipelines
    pipelines = get_pipelines(include_data_dependent=True, include_reg=True)

    # Loop over labels
    for label in labels_to_process:
        evaluate(
            fwd=fwd,
            inv=inv,
            label=label,
            pipelines=pipelines,
            metrics=["corr"],
            simulations=simulation_ids,
            simulation_path=simulation_path,
            derivative_path=derivative_path,
            fmin=params.fmin,
            fmax=params.fmax,
            sfreq=params.sfreq,
            debug=debug,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        spacing=args.spacing,
        parcellation=args.parcellation,
        simulation_name=args.simulation,
        label_name=args.label,
        debug=args.debug,
    )
