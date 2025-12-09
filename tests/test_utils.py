import numpy as np

from ctfeval.config import paths
from ctfeval.io import load_source_space
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.utils import data2stc, labeldata2data, vertno_to_index, area_to_extent


def test_data2stc():
    src = load_source_space("fsaverage", paths.subjects_dir)
    n_vertno = sum(s["nuse"] for s in src)
    data = np.arange(n_vertno)

    stc = data2stc(data, src)
    assert np.allclose(data, np.squeeze(stc.data)), "Data mismatch"
    assert np.array_equal(stc.vertices[0], src[0]["vertno"]), "Mismatch lh"
    assert np.array_equal(stc.vertices[1], src[1]["vertno"]), "Mismatch rh"


def test_labeldata2data():
    src = load_source_space("fsaverage", paths.subjects_dir)
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    n_vertno = sum(s["nuse"] for s in src)
    data_array = np.arange(p.n_labels)
    data_dict = {label.name: i for i, label in enumerate(p.labels)}

    # Check that the values are assigned correctly for one label
    test_label = p["precentral-lh"]
    test_value = p.index("precentral-lh")
    label_vertno = test_label.restrict(src).vertices
    _, ind, _ = np.intersect1d(src[0]["vertno"], label_vertno, return_indices=True)

    data_from_array = labeldata2data(data_array, p.labels, src)
    assert data_from_array.size == n_vertno
    assert np.all(data_from_array[ind] == test_value), "data from array"

    data_from_dict = labeldata2data(data_dict, p.labels, src)
    assert data_from_dict.size == n_vertno
    assert np.all(data_from_dict[ind] == test_value), "data from dict"


def test_vertno_to_index():
    src = load_source_space("fsaverage", paths.subjects_dir)
    lh_vertno = src[0]["vertno"][10]
    rh_vertno = src[0]["vertno"][20]

    assert vertno_to_index(src, "lh", lh_vertno) == 10, "lh"
    assert vertno_to_index(src, "rh", rh_vertno) == (20 + 4098), "rh"


def test_area_to_extent():
    assert area_to_extent(0.2 * 0.2 * np.pi) == 2
