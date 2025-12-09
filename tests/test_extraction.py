import mne
import numpy as np
import pytest

from ctfeval.config import paths
from ctfeval.extraction import (
    get_aggregation_weights,
    extract_label_time_course_centroid,
)
from ctfeval.io import load_source_space, load_parcellation


def test_get_aggregation_weights():
    rng = np.random.default_rng(seed=1234)
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)
    test_label = p["precentral-lh"].restrict(src)

    label_size = test_label.vertices.size
    stc = mne.SourceEstimate(
        data=rng.random((label_size, 100)),
        vertices=[test_label.vertices, []],
        tmin=0.0,
        tstep=0.01,
    )

    w_mean = get_aggregation_weights(stc, test_label, src, "mean")
    assert w_mean.size == label_size
    assert np.isclose(w_mean.sum(), 1.0)
    assert np.allclose(w_mean, 1.0 / label_size)

    w_mean_flip = get_aggregation_weights(stc, test_label, src, "mean_flip")
    assert w_mean_flip.size == label_size
    assert np.isclose(np.abs(w_mean_flip).sum(), 1.0)
    assert np.allclose(np.abs(w_mean_flip), 1.0 / label_size)

    w_pca_flip = get_aggregation_weights(stc, test_label, src, "pca_flip")
    assert w_pca_flip.size == label_size
    assert np.isclose(np.linalg.norm(w_pca_flip), 1.0)


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_extract_label_time_course_centroid(hemi):
    rng = np.random.default_rng(seed=1234)
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)
    test_label = p[f"precentral-{hemi}"].restrict(src)
    label_size = test_label.vertices.size

    hemi_idx = 1 if hemi == "rh" else 0
    center_vertno = test_label.center_of_mass(
        "fsaverage", restrict_vertices=src, subjects_dir=paths.subjects_dir
    )
    center_idx = np.where(test_label.vertices == center_vertno)

    vertices = [[], []]
    vertices[hemi_idx] = test_label.vertices
    stc = mne.SourceEstimate(
        data=rng.random((label_size, 100)), vertices=vertices, tmin=0.0, tstep=0.01
    )
    stc.data[center_idx, :] = 1.0

    label_tc, w_centroid = extract_label_time_course_centroid(
        stc, test_label, src, "fsaverage", paths.subjects_dir, return_weights=True
    )
    assert np.allclose(label_tc, 1.0)
    assert np.allclose(w_centroid.sum(), 1.0)
    assert np.isclose(w_centroid[0, center_idx], 1.0)
