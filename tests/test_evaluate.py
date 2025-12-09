import numpy as np

from ctfeval.evaluate import (
    prepare_eval_labels,
    prepare_source_cov,
    prepare_data_cov,
    get_correlation_across_rois,
)
from ctfeval.io import init_subject

from utils.prepare import dummy_simulation


def test_prepare_eval_labels():
    simulation_path, simulation_id = dummy_simulation()
    fwd, _, _, p = init_subject(parcellation="DK")

    # ROI mode = return DK labels as is
    roi_mode_labels = prepare_eval_labels(
        "roi", p.labels, simulation_path, simulation_id, fwd["src"]
    )
    assert roi_mode_labels == p.labels

    # Source mode = load from dummy simulation, point labels expected
    source_mode_labels = prepare_eval_labels(
        "source", p.labels, simulation_path, simulation_id, fwd["src"]
    )
    assert source_mode_labels != p.labels
    assert len(source_mode_labels) == p.n_labels
    assert all(len(label.vertices) == 1 for label in source_mode_labels)


def test_prepare_source_cov():
    simulation_path, simulation_id = dummy_simulation()
    fwd, _, _, p = init_subject(parcellation="DK")

    # No info
    no_info_cov = prepare_source_cov("no_info", simulation_path, simulation_id)
    assert no_info_cov is None

    # Full info
    full_info_cov = prepare_source_cov("full_info", simulation_path, simulation_id)
    assert full_info_cov is not None
    assert full_info_cov.shape == (fwd["nsource"], fwd["nsource"])


def test_prepare_data_cov():
    simulation_path, simulation_id = dummy_simulation()
    fwd, _, _, p = init_subject(parcellation="DK")

    # auto
    auto_data_cov = prepare_data_cov("auto", simulation_path, simulation_id)
    assert auto_data_cov == "auto"

    # actual
    actual_data_cov = prepare_data_cov("actual", simulation_path, simulation_id)
    assert actual_data_cov.shape == (fwd["nchan"], fwd["nchan"])


def test_get_correlation_across_rois():
    val_theory = np.vstack([np.arange(10), np.arange(10)])
    val_sim = np.vstack([np.arange(10) / 2, np.arange(10, 0, -1) - 1])

    rs, betas = get_correlation_across_rois(val_theory, val_sim)
    assert np.allclose(rs, [1, -1])
    assert np.allclose(betas, [0.5, -1])
