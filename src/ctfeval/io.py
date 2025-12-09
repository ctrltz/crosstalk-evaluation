import mne
import numpy as np

from scipy.sparse import save_npz

from ctfeval.config import paths
from ctfeval.prepare import interpolate_missing_channels, setup_inverse_operator
from ctfeval.parcellations import PARCELLATIONS


def load_head_model(subject, subjects_dir, spacing="oct6", info=None):
    # Select the forward model with desired spacing
    head_model_path = subjects_dir / subject / paths.head_model_subdir
    fwd_path = head_model_path / f"{subject}-{spacing}-fwd.fif"

    # Load forward solution, use fixed dipole orientations
    fwd = mne.read_forward_solution(fwd_path)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)

    # Pick only channels that are present in the info if provided
    if info is not None:
        fwd.pick_channels(info.ch_names, ordered=True)

    return fwd


def load_parcellation(parcellation, subject, subjects_dir):
    return PARCELLATIONS[parcellation].load(subject, subjects_dir)


def load_raw(raw_path, interpolate=False, full_info=None):
    # Read and set the EEG electrode locations, which are already in fsaverage's
    # space (MNI space) for standard_1005:
    raw = mne.io.read_raw_eeglab(raw_path, preload=True)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    if interpolate:
        assert full_info is not None, "Full Info is required for interpolation"
        raw = interpolate_missing_channels(raw, full_info)

    # Add average reference projection for inverse modeling
    raw.set_eeg_reference(projection=True)

    return raw


def load_source_space(subject, subjects_dir, spacing="oct6"):
    head_model_path = subjects_dir / subject / paths.head_model_subdir
    src_path = head_model_path / f"{subject}-{spacing}-src.fif"
    src = mne.read_source_spaces(src_path)
    return src


def init_subject(*, subject="fsaverage", spacing="oct6", parcellation=None):
    lemon_info = mne.io.read_info(paths.lemon_info)
    fwd = load_head_model(subject, paths.subjects_dir, spacing=spacing, info=lemon_info)
    inv = setup_inverse_operator(fwd, lemon_info)
    p = (
        load_parcellation(parcellation, subject, paths.subjects_dir)
        if parcellation
        else None
    )

    return fwd, inv, lemon_info, p


def get_simulation_name(
    parc_code,
    spacing,
    source_area,
    snr_mode,
    target_snr,
    var_mode,
    conn_preset,
    sensor_noise_level,
    name_extra,
):
    simulation_name = f"{parc_code}_{spacing}"
    if source_area is None:
        simulation_name += "_point"
    else:
        simulation_name += f"_patch_{source_area:.0f}cm2"
    simulation_name += f"_{snr_mode}_snr_{target_snr:.1f}_var_{var_mode}"
    simulation_name += f"_conn_{conn_preset}_sensor_noise_{sensor_noise_level:.2f}"
    if name_extra:
        simulation_name += f"_{name_extra}"

    return simulation_name


def load_simulation(simulation_path, simulation_name):
    # Load the simulated data
    raw = mne.io.read_raw(simulation_path / f"{simulation_name}_eeg.fif")

    # Set average reference as projector for inverse modeling
    raw.set_eeg_reference(projection=True)

    # Load the ground truth
    with np.load(simulation_path / f"{simulation_name}_gt.npz") as data:
        gt = data["gt"]
        names = data["names"]

    return raw, gt, names


def load_source_locations(simulation_path, simulation_name, src, source_names):
    source_labels = {}
    with np.load(simulation_path / f"{simulation_name}_source_loc.npz") as loc:
        for name in source_names:
            hemi = name[-2:]
            src_idx = 1 if hemi == "rh" else 0
            vertices = loc[name]
            source_labels[name] = mne.Label(
                vertices=vertices,
                pos=src[src_idx]["rr"][vertices, :],
                hemi=hemi,
                name=name,
            )

    return source_labels


def save_simulation(
    raw, gt, cs_source, source_loc, p, simulation_path, simulation_name, overwrite=False
):
    # Save the simulated data
    raw.save(simulation_path / f"{simulation_name}_eeg.fif", overwrite=overwrite)

    # Save the ground truth source activity
    label_names = [label.name for label in p.labels]
    np.savez(simulation_path / f"{simulation_name}_gt.npz", gt=gt, names=label_names)

    # Save the ground truth source cross-spectrum
    save_npz(simulation_path / f"{simulation_name}_cs_source.npz", cs_source)

    # Save the ground truth location of target sources
    np.savez(simulation_path / f"{simulation_name}_source_loc.npz", **source_loc)


def dict_to_npz(transform=None, **kwargs):
    result = {}
    for name, d in kwargs.items():
        if transform is not None:
            result.update({f"{name}__{k}": transform(v) for k, v in d.items()})
        else:
            result.update({f"{name}__{k}": v for k, v in d.items()})

    return result


def dict_from_npz(data, name, transform=None):
    d = {}
    prefix = f"{name}__"
    for k, v in data.items():
        if not k.startswith(prefix):
            continue

        if transform is not None:
            d[k.removeprefix(prefix)] = transform(v)
        else:
            d[k.removeprefix(prefix)] = v

    return d


def save_filters(filter_path, filters):
    method = np.array(list(set([sf.method for sf in filters])))
    assert method.size == 1

    # NOTE: if lambda is not saved explicitly for baseline approaches, use 0.05 by default
    lambdas = np.array([sf.method_params.get("lambda_", 0.05) for sf in filters])
    labels = np.array([sf.name for sf in filters])
    weights = np.vstack([sf.w for sf in filters])
    ch_names = filters[0].ch_names
    assert np.all(
        [ch_names == sf.ch_names for sf in filters]
    ), "Expected channel names to be the same for all filters"

    np.savez(
        filter_path,
        weights=weights,
        method=method,
        lambdas=lambdas,
        labels=labels,
        ch_names=ch_names,
    )
