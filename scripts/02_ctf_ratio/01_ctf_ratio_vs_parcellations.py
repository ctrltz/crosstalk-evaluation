import numpy as np

from itertools import product
from tqdm import tqdm

from roiextract.utils import get_label_mask

from ctfeval.config import paths, params
from ctfeval.ctf import get_max_ctf_ratios, get_achieved_ctf_ratios
from ctfeval.io import init_subject
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS


def main():
    # Load the head model
    fwd, inv, _, _ = init_subject()
    src = fwd["src"]

    # Source depth data
    depth = np.load(paths.precomputed / "source_depth.npy")

    # Prepare the extraction pipelines
    inv_methods = ["eLORETA"]
    roi_methods = ["mean", "mean_flip", "centroid"]
    pipelines = list(product(inv_methods, roi_methods))

    # Output folder
    save_path = paths.theory / "parcellations"
    save_path.mkdir(parents=True, exist_ok=True)

    for code, p in tqdm(PARCELLATIONS.items()):
        p.load("fsaverage", paths.subjects_dir)

        logger.info(f"Processing {p.name}")
        results = {}

        # Max CTF ratios
        max_ratios = get_max_ctf_ratios(fwd, p)
        results["max_ratios"] = max_ratios
        logger.info("[done] max ratios")

        # Compute achieved ratios only for DK and Schaefer
        if code in ["DK", "Schaefer"]:
            for inv_method, roi_method in pipelines:
                achieved_ratios = get_achieved_ctf_ratios(
                    fwd, p, inv, inv_method, roi_method, params.reg
                )
                results[f"ratios_{inv_method}_{roi_method}"] = achieved_ratios
                logger.info(f"[done] achieved ratios - {inv_method} + {roi_method}")

        # Compute parcellation features only for DK
        if code == "DK":
            # Average distance from sources to sensors
            mean_depth = np.zeros(p.n_labels)
            for i, label in enumerate(p.labels):
                mask = get_label_mask(label, src)
                mean_depth[i] = depth[mask].mean()
            results["mean_depth"] = mean_depth
            logger.info("[done] distance to sensors")

            # ROI areas
            areas_cm2 = p.areas("fsaverage", paths.subjects_dir) * 1e4
            results["areas_cm2"] = areas_cm2
            logger.info("[done] areas")

        # Save parcellation-specific results
        np.savez(save_path / f"{code}.npz", **results)
        logger.info(f"[done] {p.name}")


if __name__ == "__main__":
    main()
