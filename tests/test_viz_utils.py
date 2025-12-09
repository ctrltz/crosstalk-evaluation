import numpy as np

from ctfeval.viz_utils import prepare_colormap, sort_labels


def test_prepare_colormap_positive():
    data = np.arange(11)
    threshold = 0.3
    limits, cmap = prepare_colormap(data, threshold)

    assert limits == [0, 3, 10], "wrong limits"
    assert cmap == "Reds", "wrong colormap"


def test_prepare_colormap_negative():
    data = np.arange(-5, 6)
    threshold = 0.4
    limits, cmap = prepare_colormap(data, threshold)

    assert limits == [-5, 0, 5], "wrong limits"
    assert cmap == "RdBu_r", "wrong colormap"


def test_sort_labels():
    # avg A = 0.2, avg B = 0.3
    # -> expected order: B-lh, A-lh, A-rh, B-rh (2, 0, 1, 3)
    label_names = ["A-lh", "A-rh", "B-lh", "B-rh"]
    ratios = [0.1, 0.3, 0.2, 0.4]
    expected_order = np.array([2, 0, 1, 3])
    expected_values = np.array([0.2, 0.2, 0.3, 0.3])

    order = sort_labels(label_names, ratios)
    assert np.array_equal(order, expected_order)

    values = sort_labels(label_names, ratios, return_values=True)
    assert np.allclose(values, expected_values)
