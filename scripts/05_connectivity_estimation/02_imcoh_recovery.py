import argparse
import numpy as np

from roiextract.filter import SpatialFilter
from roiextract.utils import get_label_mask
from tqdm import tqdm

from ctfeval.config import paths, params, get_roi_methods
from ctfeval.io import init_subject


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=get_roi_methods(False), required=True)


def get_imcoh_recovery(fwd, inv, labels, inv_method, roi_method):
    src = fwd["src"]
    n_sources = fwd["nsource"]
    L = fwd["sol"]["data"]
    filters = []

    n_labels = len(labels)
    ctf_all = np.zeros((n_labels, n_sources))
    for i_label, label in enumerate(tqdm(labels)):
        sf = SpatialFilter.from_inverse(
            fwd,
            inv,
            label,
            inv_method,
            params.reg,
            roi_method,
            "fsaverage",
            paths.subjects_dir,
        )
        ctf = sf.w @ L
        ctf /= np.sqrt(ctf @ ctf.T)

        filters.append(sf)
        ctf_all[i_label, :] = ctf

    recovery = np.zeros((n_labels, n_labels, n_labels, n_labels))
    for i_rec1, ctf1 in enumerate(tqdm(ctf_all, desc="outer loop")):
        for i_rec2, ctf2 in enumerate(tqdm(ctf_all, desc=f"inner loop #{i_rec1}")):
            ctf1 = np.atleast_2d(ctf1)
            ctf2 = np.atleast_2d(ctf2)
            rec = np.abs(ctf1.T @ ctf2 - ctf2.T @ ctf1)

            # Get average weight of ImCoh contribution for all pairs of sources
            # located in ROIs i_gt1 and i_gt2 to the estimated ImCoh between
            # time series of ROIs i_rec1 and i_rec2
            for i_gt1, label_gt1 in enumerate(labels):
                for i_gt2, label_gt2 in enumerate(labels):
                    idx1 = np.where(get_label_mask(label_gt1, src))[0]
                    idx2 = np.where(get_label_mask(label_gt2, src))[0]
                    recovery[i_rec1, i_rec2, i_gt1, i_gt2] = rec[
                        np.ix_(idx1, idx2)
                    ].mean()

    return recovery


def main(roi_method):
    fwd, inv, _, p = init_subject(parcellation="DK")
    recovery = get_imcoh_recovery(fwd, inv, p.labels, "eLORETA", roi_method)

    output_path = paths.derivatives / "theory" / "connectivity_estimation"
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / f"recovery_{roi_method}.npy", recovery)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.method)
