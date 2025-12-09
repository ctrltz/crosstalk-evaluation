import argparse
import json
import numpy as np

from ctfeval.config import paths, get_pipelines
from ctfeval.io import init_subject
from ctfeval.parcellations import PARCELLATIONS


parser = argparse.ArgumentParser()
parser.add_argument("--spacing", choices=["oct6", "ico4"], default="oct6")
parser.add_argument("--parcellation", choices=PARCELLATIONS.keys(), default="DK")
parser.add_argument("--simulation", type=str, required=True)
parser.add_argument("--method", type=str, required=True)


def main(spacing, parcellation, simulation_name, method_name):
    # Prepare the head model
    fwd, _, _, p = init_subject(spacing=spacing, parcellation=parcellation)

    simulation_path = paths.simulations / simulation_name
    results_path = paths.derivatives / "simulations" / simulation_name / method_name
    save_path = (
        paths.derivatives / "simulations" / simulation_name / method_name / "combined"
    )
    save_path.mkdir(exist_ok=True)

    with open(simulation_path / "simulations.json") as f:
        simulation_desc = json.load(f)
    n_simulations = len(simulation_desc["filenames"])

    pipelines = get_pipelines(include_data_dependent=True, include_reg=True)
    n_pipelines = len(pipelines)

    # Collect the results of simulations
    corr = np.zeros((n_pipelines, p.n_labels, n_simulations))
    filters = np.zeros((n_pipelines, p.n_labels, n_simulations, fwd["nchan"]))
    for i_label, label in enumerate(p.labels):
        with np.load(results_path / f"results_{label.name}.npz") as data:
            filters[:, i_label, :, :] = data["filters"][:, :n_simulations, :]
            saved_pipelines = data["pipelines"]
            assert np.array_equal(np.array(pipelines), saved_pipelines)
            corr[:, i_label, :] = data["corr"]

    np.save(save_path / "filters.npy", filters)
    np.save(save_path / "pipelines.npy", pipelines)
    np.save(save_path / "results.npy", corr)


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        spacing=args.spacing,
        parcellation=args.parcellation,
        simulation_name=args.simulation,
        method_name=args.method,
    )
