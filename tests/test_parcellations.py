import pytest

from ctfeval.config import paths
from ctfeval.parcellations import PARCELLATIONS


def test_parcellation_load():
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)

    assert p.n_labels == 68
    assert len(p.label_names) == p.n_labels


@pytest.mark.parametrize("label_name", ["postcentral-rh", "insula-rh"])
def test_parcellation_getitem(label_name):
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    assert p[label_name].name == label_name


@pytest.mark.parametrize("label_name", ["postcentral-lh", "insula-lh"])
def test_parcellation_index(label_name):
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)
    assert p.labels[p.index(label_name)].name == label_name
