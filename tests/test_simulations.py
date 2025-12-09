import numpy as np

from ctfeval.config import paths, params
from ctfeval.io import load_source_space, load_parcellation
from ctfeval.simulations import (
    select_random_in_labels,
    grow_patches_in_labels,
    extract_ground_truth,
    extract_source_locations,
    extract_source_cs,
)

from utils.prepare import dummy_simulation


def test_select_random_in_labels():
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)

    test_labels = [p.labels[i] for i in [0, 10, 20, 30, 40, 50]]
    vertices = select_random_in_labels(src, labels=test_labels)

    for v, label in zip(vertices, test_labels):
        src_idx, vertno = v
        hemi = "rh" if src_idx else "lh"
        assert hemi == label.hemi, "Wrong hemi"
        assert vertno in list(label.vertices), "Vertex does not belong to the label"


def test_select_random_in_labels_random_state():
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)

    test_labels = [p.labels[i] for i in [0, 10, 20, 30, 40, 50]]
    vertices_123 = select_random_in_labels(src, labels=test_labels, random_state=123)
    vertices_456 = select_random_in_labels(src, labels=test_labels, random_state=456)
    assert vertices_123 != vertices_456, "Expected random selections"

    vertices_123_rerun = select_random_in_labels(
        src, labels=test_labels, random_state=123
    )
    assert vertices_123 == vertices_123_rerun, "Expected reproducibility"


def test_grow_patches_in_labels():
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)

    test_labels = [p.labels[i] for i in [0, 10, 20, 30, 40, 50]]
    patches = grow_patches_in_labels(
        src,
        extent=10,
        labels=test_labels,
        subjects_dir=paths.subjects_dir,
        random_state=123,
    )

    for patch, label in zip(patches, test_labels):
        src_idx, vertno = patch
        hemi = "rh" if src_idx else "lh"
        assert hemi == label.hemi, "Wrong hemi"

        common = np.intersect1d(vertno, label.vertices)
        assert len(common) == len(vertno), "Patch vertices do not belong to the label"


def test_grow_patches_in_labels_random_state():
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)

    test_labels = [p.labels[i] for i in [0, 10, 20, 30, 40, 50]]
    vertices_123 = grow_patches_in_labels(
        src,
        extent=10,
        labels=test_labels,
        subjects_dir=paths.subjects_dir,
        random_state=123,
    )
    vertices_456 = grow_patches_in_labels(
        src,
        extent=10,
        labels=test_labels,
        subjects_dir=paths.subjects_dir,
        random_state=456,
    )
    assert vertices_123 != vertices_456, "Expected random selections"

    vertices_123_rerun = grow_patches_in_labels(
        src,
        extent=10,
        labels=test_labels,
        subjects_dir=paths.subjects_dir,
        random_state=123,
    )
    assert vertices_123 == vertices_123_rerun, "Expected reproducibility"


def test_ground_truth():
    """
    Combining all ground-truth extraction in one test.
    """
    _, _, sc = dummy_simulation(return_sc=True)
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = load_parcellation("DK", "fsaverage", paths.subjects_dir)

    # Source time courses
    gt = extract_ground_truth(sc, p.label_names)
    assert gt.shape[0] == p.n_labels, "Wrong number of sources"
    assert gt.shape[1] == len(sc.times), "Wrong number of samples"
    for i, name in enumerate(p.label_names):
        assert np.allclose(gt[i, :], sc[name].waveform)

    # Source locations
    source_loc = extract_source_locations(sc, p.label_names)
    assert isinstance(source_loc, dict)
    for name in p.label_names:
        assert np.array_equal(source_loc[name], sc[name].to_label(src).vertices)

    # Source cross-spectra
    source_cs = extract_source_cs(sc, params.fmin, params.fmax)
    stc = sc.to_stc()
    n_vertices = len(stc.vertices[0]) + len(stc.vertices[1])
    assert source_cs.nnz == n_vertices * n_vertices
