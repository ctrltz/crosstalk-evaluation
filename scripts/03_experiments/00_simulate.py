import argparse
import numpy as np
import json

from datetime import datetime

from tqdm import tqdm
from zlib import crc32

from ctfeval.config import paths, params
from ctfeval.io import get_simulation_name, save_simulation, init_subject
from ctfeval.log import logger
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.simulations import (
    simulation_activity_reconstruction,
    extract_ground_truth,
    extract_source_cs,
    extract_source_locations,
    get_source_variance,
)
from ctfeval.utils import data2stc


parser = argparse.ArgumentParser()
parser.add_argument(
    "--spacing",
    choices=["oct6", "ico4", "ico5"],
    default="oct6",
    help="The source space grid to use for simulations",
)
parser.add_argument(
    "--parcellation",
    choices=PARCELLATIONS.keys(),
    required=True,
    help="Parcellation to use for placing sources",
)
parser.add_argument(
    "--snr-mode",
    choices=["local", "global"],
    default="global",
    help="The mode to use when setting up the SNR",
)
parser.add_argument(
    "--target-snr",
    type=float,
    default=params.target_snr,
    help="Target global signal-to-noise ratio (alpha vs. 1/f)",
)
parser.add_argument(
    "--var-mode",
    type=str,
    default="equal",
    help="Can be 'equal' or the name of the file with variance values",
)
parser.add_argument(
    "--source-type",
    choices=["point", "patch"],
    default="point",
    help="The type of sources (point-like or cortical patches)",
)
parser.add_argument("--patch-area", type=float, default=None, help="Patch area in cm2")
parser.add_argument(
    "--conn-preset",
    choices=["none", "weak", "strong"],
    default="none",
    help="Preset for setting up ground truth connectivity",
)
parser.add_argument(
    "--sensor-noise-level",
    type=float,
    default=params.sensor_noise_level,
    help="The level of sensor noise",
)
parser.add_argument(
    "-n",
    type=int,
    default=params.n_simulations,
    help="The number of datasets to simulate",
)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument(
    "--make-plots", action="store_true", help="Plot the generated source configurations"
)
parser.add_argument(
    "--extra", type=str, default="", help="Extra comments for the simulation name"
)


def simulate(
    spacing,
    parcellation,
    snr_mode,
    target_snr,
    var_mode,
    source_type,
    patch_area,
    conn_preset,
    sensor_noise_level,
    n_simulations,
    overwrite,
    make_plots,
    name_extra,
):
    # Resolve the provided parameters
    preset = params.connectivity[conn_preset]
    n_connections = preset["n_connections"]
    mean_coherence = preset["mean_coherence"]
    std_coherence = preset["std_coherence"]
    cfg = dict(
        spacing=spacing,
        parcellation=parcellation,
        snr_mode=snr_mode,
        target_snr=target_snr,
        var_mode=var_mode,
        source_type=source_type,
        patch_area=patch_area,
        conn_preset=conn_preset,
        n_connections=n_connections,
        mean_coherence=mean_coherence,
        std_coherence=std_coherence,
        sensor_noise_level=sensor_noise_level,
        n_simulations=n_simulations,
        overwrite=overwrite,
        make_plots=make_plots,
        name_extra=name_extra,
    )
    logger.info(f"Provided parameters: {json.dumps(cfg, indent=4)}")

    assert spacing in ["oct6", "ico4", "ico5"]

    # Load the head model
    fwd, _, lemon_info, p = init_subject(spacing=spacing, parcellation=parcellation)
    src = fwd["src"]

    # Create a folder for the simulated data
    simulation_name = get_simulation_name(
        p.code,
        spacing,
        patch_area,
        snr_mode,
        target_snr,
        var_mode,
        conn_preset,
        sensor_noise_level,
        name_extra,
    )
    simulation_path = paths.simulations / simulation_name
    simulation_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving the simulated data to {simulation_path}")

    # Prepare the std
    if var_mode == "equal":
        std = 1
    else:
        var_filename = paths.precomputed / f"var_{spacing}_{var_mode}.npy"
        source_var = np.load(var_filename)
        std = data2stc(np.sqrt(source_var), src)

    # Modify the seed of the random generator for each simulation
    # in a controllable way by adding the hash function of the simulation name
    main_seed = params.seed + crc32(simulation_name.encode("utf-8"))

    ss = np.random.SeedSequence(main_seed)
    seeds = ss.generate_state(n_simulations)
    source_var = np.full((fwd["nsource"], n_simulations), np.nan)
    for i_run, seed in enumerate(tqdm(seeds)):
        # Define the simulation
        sim = simulation_activity_reconstruction(
            fwd=fwd,
            labels=p.labels,
            n_noise_dipoles=params.n_noise_dipoles,
            target_snr=target_snr,
            snr_mode=snr_mode,
            std=std,
            source_type=source_type,
            patch_area_cm2=patch_area,
            n_connections=n_connections,
            mean_coherence=mean_coherence,
            std_coherence=std_coherence,
            fmin=params.fmin,
            fmax=params.fmax,
            random_state=seed,
        )

        # Simulate the data
        snr_global = target_snr if snr_mode == "global" else None
        sc = sim.simulate(
            sfreq=params.sfreq,
            duration=params.duration,
            fwd=fwd,
            snr_global=snr_global,
            snr_params=dict(fmin=params.fmin, fmax=params.fmax),
            random_state=seed,
        )

        output_name = f"sim_{i_run:04d}_{seed}"
        raw = sc.to_raw(fwd, lemon_info, sensor_noise_level=sensor_noise_level)
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
            output_name,
            overwrite=overwrite,
        )

        # Store the source variance to check that adjustment worked
        source_var[:, i_run] = get_source_variance(sc, fwd)

    metadata = {
        "created": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
        "seed": str(main_seed),
        "filenames": [f"sim_{i_run:04d}_{seed}" for i_run, seed in enumerate(seeds)],
    }
    metadata.update(cfg)
    with open(simulation_path / "simulations.json", "w+") as f:
        json.dump(metadata, f, indent=4)

    # Save the source variance
    np.save(simulation_path / "source_var.npy", source_var)


if __name__ == "__main__":
    args = parser.parse_args()

    simulate(
        spacing=args.spacing,
        parcellation=args.parcellation,
        snr_mode=args.snr_mode,
        target_snr=args.target_snr,
        var_mode=args.var_mode,
        source_type=args.source_type,
        patch_area=args.patch_area,
        conn_preset=args.conn_preset,
        sensor_noise_level=args.sensor_noise_level,
        n_simulations=args.n,
        overwrite=args.overwrite,
        make_plots=args.make_plots,
        name_extra=args.extra,
    )
