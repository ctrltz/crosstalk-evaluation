import argparse
import pandas as pd

from itertools import product

from ctfeval.config import paths
from ctfeval.datasets import rift_subfolder
from ctfeval.log import logger
from ctfeval.rift.metrics import METRICS


parser = argparse.ArgumentParser()
parser.add_argument("--tagging-type", required=True)
parser.add_argument("--random-phases", required=True)
parser.add_argument("--measure", choices=["coh", "imcoh"], required=True)
parser.add_argument("--kind", choices=["brain_stimulus", "brain_brain"], required=True)
parser.add_argument("--onestim", action="store_true")


def main(tagging_type, random_phases, measure, kind, onestim):
    # Paths
    tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
    results_path = tag_folder / kind

    # Models to be compared
    name_template = "search_results_{0}.csv"
    models = ["no_leakage", "distance", "ctf"]

    # Load all results
    dfs = []

    for model in models:
        model_filename = name_template.format(model)
        logger.info(model_filename)

        df_model = pd.read_csv(results_path / model_filename)
        df_model["model"] = model
        dfs.append(df_model)

    df = pd.concat(dfs)
    filename_combined = "combined_results.csv"
    df.to_csv(
        results_path / filename_combined,
        index=False,
    )
    logger.info("Loaded all results")

    # Pick the best result model-wise for each metric
    for metric, option, target in product(
        METRICS.keys(), ["raw", "delta"], ["fit", "all"]
    ):
        metric_full = f"{metric}_{option}_{target}"

        best_row_per_model = df.sort_values(
            metric_full, ascending=False
        ).drop_duplicates(["model"])
        best_row_per_model.to_csv(
            results_path / f"best_results_{metric}_{option}_{target}.csv",
            index=False,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        tagging_type=int(args.tagging_type),
        random_phases=int(args.random_phases),
        measure=args.measure,
        kind=args.kind,
        onestim=args.onestim,
    )
