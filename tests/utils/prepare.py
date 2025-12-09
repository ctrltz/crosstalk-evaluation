from pathlib import Path

from ctfeval.config import params
from ctfeval.io import init_subject, save_simulation
from ctfeval.simulations import (
    simulation_activity_reconstruction,
    extract_ground_truth,
    extract_source_cs,
    extract_source_locations,
)


def dummy_simulation(return_sc=False):
    """
    A dummy simulation for testing the rest of the workflow.
    """
    simulation_path = Path.cwd() / "tests" / "data"
    simulation_path.mkdir(parents=True, exist_ok=True)
    simulation_id = "dummy"
    raw_path = simulation_path / f"{simulation_id}_eeg.fif"

    if raw_path.exists() and not return_sc:
        return simulation_path, simulation_id

    # Load the head model
    fwd, _, lemon_info, p = init_subject(parcellation="DK")

    sim = simulation_activity_reconstruction(
        fwd=fwd,
        labels=p.labels,
        n_noise_dipoles=params.n_noise_dipoles,
        target_snr=params.target_snr,
        snr_mode=params.snr_mode,
        std=1,
        source_type="point",
        patch_area_cm2=None,
        n_connections=10,
        mean_coherence=0.25,
        std_coherence=0.1,
        fmin=params.fmin,
        fmax=params.fmax,
        random_state=params.seed,
    )

    # Simulate the data
    sc = sim.simulate(
        sfreq=params.sfreq,
        duration=params.duration,
        fwd=fwd,
        snr_global=params.target_snr,
        snr_params=dict(fmin=params.fmin, fmax=params.fmax),
        random_state=params.seed,
    )

    raw = sc.to_raw(fwd, lemon_info, sensor_noise_level=params.sensor_noise_level)
    label_names = [label.name for label in p.labels]
    gt = extract_ground_truth(sc, label_names)
    cs_source = extract_source_cs(sc, params.fmin, params.fmax)
    source_loc = extract_source_locations(sc, p.label_names)

    save_simulation(
        raw,
        gt,
        cs_source,
        source_loc,
        p,
        simulation_path,
        simulation_id,
        overwrite=True,
    )

    if not return_sc:
        return simulation_path, simulation_id

    return simulation_path, simulation_id, sc
