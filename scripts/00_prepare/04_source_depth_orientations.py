import numpy as np
import matplotlib.pyplot as plt
import mne

from mne.io.constants import FIFF
from mne.surface import _project_onto_surface, _read_mri_surface
from mne.transforms import _get_trans
from scipy.linalg import norm

from ctfeval.config import paths
from ctfeval.io import init_subject
from ctfeval.viz import plot_data_on_brain


def estimate(src, surf):
    # Project sources onto the outer surface of the skull
    rrs_orig = np.vstack([s["rr"][s["inuse"] > 0, :] for s in src])
    nns_orig = np.vstack([s["nn"][s["inuse"] > 0, :] for s in src])
    _, _, rrs_proj, nns_proj = _project_onto_surface(
        rrs_orig, surf, project_rrs=True, return_nn=True, method="nearest"
    )

    # Calculate dot product between dipole orientations and corresponding normals
    # to the outer skull
    n_dipoles = nns_orig.shape[0]
    dp = np.zeros((n_dipoles,))
    depth = np.zeros((n_dipoles,))
    for i in range(n_dipoles):
        nn1 = nns_orig[i, :]
        nn2 = nns_proj[i, :]
        dp[i] = np.dot(nn1, nn2) / (norm(nn1) * norm(nn2))

        rr1 = rrs_orig[i, :]
        rr2 = rrs_proj[i, :]
        depth[i] = np.sqrt(np.sum((rr1 - rr2) ** 2))

    return dp, depth


def main(spacing="oct6", surface="outer_skull", make_plots=True):
    # Load the head model
    fwd, *_ = init_subject(spacing=spacing)
    src = fwd["src"]

    # Load the surface that is used to calculate the distances to
    subject = "fsaverage"
    head_model_dir = paths.subjects_dir / subject / paths.head_model_subdir
    surf = _read_mri_surface(head_model_dir / f"{surface}.surf")

    # Transform MRI surfaces into head coordinates (source space should be in head coordinates)
    assert (
        src[0]["coord_frame"] == FIFF.FIFFV_COORD_HEAD
    ), "Expected head coordinate frame for the source spaces"
    trans = mne.read_trans(head_model_dir / f"{subject}-trans.fif")
    mri_head_t = _get_trans(trans, "mri", "head")[0]
    surf_rr_head = mne.transforms.apply_trans(mri_head_t, surf["rr"], move=True)
    surf_trans = surf.copy()
    surf_trans["rr"] = surf_rr_head

    # Estimate source depth and orientations (dot product to the normal to the skull surface)
    dp, depth = estimate(src, surf_trans)

    # Save the results
    output_path = paths.derivatives / "precomputed"
    output_path.mkdir(exist_ok=True, parents=True)
    np.save(paths.precomputed / "source_depth.npy", depth)
    np.save(paths.precomputed / "source_ori.npy", dp)

    if not make_plots:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_data_on_brain(
        np.abs(dp),
        src,
        subject,
        surface="pial_semi_inflated",
        cortex="low_contrast",
        background="white",
        colormap="RdBu_r",
        hemi="split",
        views=["lat", "med"],
        make_screenshot=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(
        paths.sanity / f"source_orientations_{spacing}.png", bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_data_on_brain(
        depth,
        src,
        subject,
        surface="pial_semi_inflated",
        cortex="low_contrast",
        background="white",
        hemi="split",
        views=["lat", "med"],
        make_screenshot=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(paths.sanity / f"source_depth_{spacing}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(make_plots=False)
