import numpy as np
import pytest

from ctfeval.config import paths
from ctfeval.io import load_source_space
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.rift.models import (
    NoSpreadModel,
    DistanceSpreadModel,
    CTFSpreadModel,
    rift_source_cov,
    normalize_ctf,
)
from ctfeval.utils import vertno_to_index


@pytest.mark.parametrize(
    "lh_label_name, rh_label_name",
    [
        ("pericalcarine-lh", "pericalcarine-rh"),
        ("precentral-lh", "precentral-rh"),
    ],
)
def test_rift_no_spread_model(lh_label_name, rh_label_name):
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p[lh_label_name]
    rh_label = p[rh_label_name]

    # Pick a vertex from both labels
    lh_vertno = lh_label.vertices[0]
    rh_vertno = rh_label.vertices[0]

    model = NoSpreadModel(p.labels, lh_vertno, rh_vertno)

    lh_idx = p.index(lh_label_name)
    rh_idx = p.index(rh_label_name)

    coh_onestim = model.pred_onestim()
    assert coh_onestim.sum() == 2, "onestim: total count"
    assert coh_onestim[lh_idx], "onestim: left hemi"
    assert coh_onestim[rh_idx], "onestim: right hemi"

    imcoh_twostim = model.pred_twostim()
    assert imcoh_twostim.sum() == 2, "twostim: total count"
    assert imcoh_twostim[lh_idx, rh_idx], "twostim: left hemi first"
    assert imcoh_twostim[rh_idx, lh_idx], "twostim: right hemi first"


@pytest.mark.parametrize(
    "lh_label_name, rh_label_name",
    [
        ("pericalcarine-lh", "pericalcarine-rh"),
        ("frontalpole-lh", "frontalpole-rh"),
    ],
)
def test_rift_distance_model(lh_label_name, rh_label_name):
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p[lh_label_name]
    rh_label = p[rh_label_name]

    # Pick a vertex from both labels
    lh_vertno = lh_label.vertices[0]
    rh_vertno = rh_label.vertices[0]

    model = DistanceSpreadModel(p.labels, src, lh_vertno, rh_vertno)

    lh_idx = p.index(lh_label_name)
    rh_idx = p.index(rh_label_name)

    # Distance should be the smallest for ROIs that actually contain SSVEP generators
    # NOTE: this test won't for some ROIs (e.g., precentral ones) due to their geometry
    coh_onestim = model.pred_onestim()
    assert lh_idx in set(np.argsort(coh_onestim)[:2]), "onestim: lh in argmax"
    assert rh_idx in set(np.argsort(coh_onestim)[:2]), "onestim: rh in argmax"

    imcoh_twostim = model.pred_twostim()
    imcoh_lh_rh = imcoh_twostim[lh_idx, rh_idx]
    assert imcoh_lh_rh in set(
        np.sort(imcoh_twostim.ravel())[:8]
    ), "twostim: (lh, rh) among the smallest"


@pytest.mark.parametrize(
    "lh_label_name, rh_label_name, gamma",
    [
        ("pericalcarine-lh", "pericalcarine-rh", 1),
        ("precentral-lh", "precentral-rh", 10),
    ],
)
def test_rift_ctf_model(lh_label_name, rh_label_name, gamma):
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p[lh_label_name]
    rh_label = p[rh_label_name]

    # Pick a vertex from both labels
    lh_vertno = lh_label.restrict(src).vertices[0]
    rh_vertno = rh_label.restrict(src).vertices[0]

    model = CTFSpreadModel(p.labels, src, lh_vertno, rh_vertno, gamma)

    lh_idx = vertno_to_index(src, "lh", lh_vertno)
    rh_idx = vertno_to_index(src, "rh", rh_vertno)

    # Set CTF to zero for all vertices expect generators, gamma should not play a role
    n_filters = 2
    n_sources = sum(s["nuse"] for s in src)
    ctfs = np.zeros((n_filters, n_sources))
    ctfs[0, lh_idx] = 1.5
    ctfs[0, rh_idx] = -0.5
    ctfs[1, lh_idx] = ctfs[1, rh_idx] = 1.0

    # Regardless of the CTF weights, coherence should be 1 unless signals cancel out
    # completely
    coh_onestim = model.pred_onestim(ctfs)
    assert np.isclose(coh_onestim[0], 1.0)
    assert np.isclose(coh_onestim[1], 1.0)

    imcoh_twostim = model.pred_twostim(ctfs)
    imcoh_expected = np.array([[0, 2.0 / np.sqrt(5.0)], [2.0 / np.sqrt(5.0), 0]])
    assert np.allclose(imcoh_twostim, imcoh_expected)


def test_rift_ctf_model_gamma():
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p["pericalcarine-lh"]
    rh_label = p["pericalcarine-rh"]

    # Pick a vertex from both labels
    lh_vertno = lh_label.restrict(src).vertices[0]
    rh_vertno = rh_label.restrict(src).vertices[0]

    lh_idx = vertno_to_index(src, "lh", lh_vertno)
    rh_idx = vertno_to_index(src, "rh", rh_vertno)

    # Set CTF to one for all vertices expect generators, set gamma to get a nice
    # number in the denominator
    n_filters = 2
    n_sources = sum(s["nuse"] for s in src)
    ctfs = np.ones((n_filters, n_sources))
    ctfs[0, lh_idx] = 1.5
    ctfs[0, rh_idx] = -0.5
    ctfs[1, lh_idx] = ctfs[1, rh_idx] = 1.0

    model = CTFSpreadModel(
        p.labels, src, lh_vertno, rh_vertno, gamma=(n_sources - 2) / 8
    )
    coh_onestim = model.pred_onestim(ctfs)
    assert np.isclose(coh_onestim[0], 1.0 / 3.0)
    assert np.isclose(coh_onestim[1], 1.0 / np.sqrt(3.0))

    model = CTFSpreadModel(
        p.labels, src, lh_vertno, rh_vertno, gamma=(n_sources - 2) / 2
    )
    imcoh_twostim = model.pred_twostim(ctfs)
    imcoh_expected = np.array([[0, 1.0 / np.sqrt(4.5)], [1.0 / np.sqrt(4.5), 0]])
    assert np.allclose(imcoh_twostim, imcoh_expected)


@pytest.mark.parametrize(
    "lh_label_name, rh_label_name",
    [
        ("pericalcarine-lh", "pericalcarine-rh"),
        ("frontalpole-lh", "frontalpole-rh"),
    ],
)
def test_rift_source_cov(lh_label_name, rh_label_name):
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p[lh_label_name]
    rh_label = p[rh_label_name]

    # Pick a vertex from both labels
    lh_vertno = lh_label.restrict(src).vertices[0]
    rh_vertno = rh_label.restrict(src).vertices[0]

    # Onestim
    source_cov_onestim, lh_idx, rh_idx = rift_source_cov(src, lh_vertno, rh_vertno)
    assert np.allclose(np.diag(source_cov_onestim), 1.0), "onestim: diagonal"
    assert np.isclose(
        source_cov_onestim[lh_idx, rh_idx], 1.0
    ), "onestim: off-diagonal, lh first"
    assert np.isclose(
        source_cov_onestim[rh_idx, lh_idx], 1.0
    ), "onestim: off-diagonal, rh first"

    # Twostim
    source_cov_twostim, lh_idx, rh_idx = rift_source_cov(
        src, lh_vertno, rh_vertno, onestim=False
    )
    assert np.allclose(np.diag(source_cov_twostim), 1.0), "onestim: diagonal"
    assert np.isclose(
        source_cov_twostim[lh_idx, rh_idx], 0.0
    ), "onestim: off-diagonal, lh first"
    assert np.isclose(
        source_cov_twostim[rh_idx, lh_idx], 0.0
    ), "onestim: off-diagonal, rh first"


@pytest.mark.parametrize("gamma", [10, 25, 100])
def test_rift_source_cov_gamma(gamma):
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p["pericalcarine-lh"]
    rh_label = p["pericalcarine-rh"]

    # Pick a vertex from both labels
    lh_vertno = lh_label.restrict(src).vertices[0]
    rh_vertno = rh_label.restrict(src).vertices[0]

    # Onestim
    source_cov_onestim, lh_idx, rh_idx = rift_source_cov(
        src, lh_vertno, rh_vertno, gamma=gamma
    )
    mask = np.full(source_cov_onestim.shape[0], True)
    mask[lh_idx] = mask[rh_idx] = False
    assert np.isclose(source_cov_onestim[lh_idx, lh_idx], 1.0), "onestim: diagonal, lh"
    assert np.isclose(source_cov_onestim[rh_idx, rh_idx], 1.0), "onestim: diagonal, rh"
    assert np.allclose(
        np.diag(source_cov_onestim)[mask], 1.0 / gamma
    ), "onestim: diagonal, other"

    # Twostim
    source_cov_twostim, lh_idx, rh_idx = rift_source_cov(
        src, lh_vertno, rh_vertno, gamma=gamma, onestim=False
    )
    assert np.isclose(
        source_cov_twostim[lh_idx, rh_idx], 0.0
    ), "onestim: off-diagonal, lh first"
    assert np.isclose(
        source_cov_twostim[rh_idx, lh_idx], 0.0
    ), "onestim: off-diagonal, rh first"
    assert np.allclose(
        np.diag(source_cov_twostim)[mask], 1.0 / gamma
    ), "onestim: diagonal, other"


@pytest.mark.parametrize("onestim", [True, False])
def test_normalize_ctf(onestim):
    """
    Cross-test normalization approaches against each other.
    """
    src = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    lh_label = p["pericalcarine-lh"]
    rh_label = p["pericalcarine-rh"]

    # Pick a vertex from both labels
    lh_vertno = lh_label.restrict(src).vertices[0]
    rh_vertno = rh_label.restrict(src).vertices[0]

    # Generate random ctfs
    n_filters = 10
    n_sources = sum(s["nuse"] for s in src)
    ctfs = np.random.randn(n_filters, n_sources)

    # Cross-test
    source_cov, lh_idx, rh_idx = rift_source_cov(
        src, lh_vertno, rh_vertno, gamma=10, onestim=onestim
    )
    ctf_loops = normalize_ctf(ctfs, source_cov, lh_idx, rh_idx, approach="loops")
    ctf_matmul = normalize_ctf(ctfs, source_cov, lh_idx, rh_idx, approach="matmul")
    ctf_fast = normalize_ctf(ctfs, source_cov, lh_idx, rh_idx, approach="fast")

    assert np.allclose(ctf_fast, ctf_loops), "fast <-> loops"
    assert np.allclose(ctf_fast, ctf_matmul), "fast <-> matmul"
