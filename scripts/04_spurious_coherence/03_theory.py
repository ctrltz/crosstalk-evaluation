import numpy as np

from tqdm import tqdm

from roiextract.filter import SpatialFilter

from ctfeval.config import paths, params
from ctfeval.io import init_subject
from ctfeval.ctf import spurious_coherence


def main():
    fwd, inv, _, p = init_subject(parcellation="DK")
    n_sources = fwd["nsource"]

    inv_method = "eLORETA"
    roi_methods = ["mean", "mean_flip", "centroid"]

    # Get CTFs for all methods
    ctfs = {}
    for roi_method in roi_methods:
        ctfs[roi_method] = np.zeros((p.n_labels, n_sources))

        for i_label, label in enumerate(tqdm(p.labels, desc=f"CTF - {roi_method}")):
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
            ctf = sf.get_ctf_fwd(fwd, mode="amplitude", normalize=None)
            ctfs[roi_method][i_label, :] = np.squeeze(ctf.data)

    # Load source covariance matrices
    source_cov_EO = np.diag(np.load(paths.precomputed / "var_oct6_EO_like.npy"))
    source_cov_EC = np.diag(np.load(paths.precomputed / "var_oct6_EC_like.npy"))

    # Estimate spurious coherence based on CTF
    sc_equal = {}
    sc_EO = {}
    sc_EC = {}

    for roi_method in roi_methods:
        sc_equal[roi_method] = np.zeros((p.n_labels, p.n_labels))
        sc_EO[roi_method] = np.zeros((p.n_labels, p.n_labels))
        sc_EC[roi_method] = np.zeros((p.n_labels, p.n_labels))

        for i1, ctf1 in enumerate(tqdm(ctfs[roi_method], desc=roi_method)):
            for i2, ctf2 in enumerate(ctfs[roi_method]):
                sc_eq_12 = spurious_coherence(ctf1, ctf2)
                sc_equal[roi_method][i1, i2] = sc_eq_12

                sc_EO_12 = spurious_coherence(ctf1, ctf2, source_cov_EO)
                sc_EO[roi_method][i1, i2] = sc_EO_12

                sc_EC_12 = spurious_coherence(ctf1, ctf2, source_cov_EC)
                sc_EC[roi_method][i1, i2] = sc_EC_12

    # Save the result
    output_path = paths.derivatives / "theory" / "spurious_coherence"
    output_path.mkdir(exist_ok=True, parents=True)
    np.savez(output_path / "sc_equal.npz", methods=np.array(roi_methods), **sc_equal)
    np.savez(output_path / "sc_EO.npz", methods=np.array(roi_methods), **sc_EO)
    np.savez(output_path / "sc_EC.npz", methods=np.array(roi_methods), **sc_EC)


if __name__ == "__main__":
    main()
