from ctfeval.config import get_inverse_methods, get_roi_methods, get_pipelines


def test_get_inverse_methods():
    assert get_inverse_methods(False) == ["eLORETA"]


def test_get_roi_methods():
    # The order is important (e.g., for visualization) so we test it here
    assert get_roi_methods(False) == ["mean", "mean_flip", "centroid"]


def test_get_pipelines():
    assert len(get_pipelines(False)) == 3
    assert [el[1] for el in get_pipelines(False)] == ["mean", "mean_flip", "centroid"]


def test_get_pipelines_reg():
    pipelines = get_pipelines(True, True)

    assert len(pipelines) == 8
    assert all(len(el) == 3 for el in pipelines)
    assert all(el[1] == 0.05 for el in pipelines)
