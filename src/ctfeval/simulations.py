import mne
import networkx as nx
import numpy as np

from itertools import combinations
from scipy.signal import filtfilt, butter
from scipy.sparse import csr_matrix

from meegsim.coupling import ppc_shifted_copy_with_noise
from meegsim.location import select_random
from meegsim.simulate import SourceSimulator
from meegsim.waveform import narrowband_oscillation
from meegsim.utils import normalize_variance

from ctfeval.config import paths, params
from ctfeval.connectivity import data2cs_hilbert
from ctfeval.log import logger
from ctfeval.prepare import prepare_filter
from ctfeval.utils import area_to_extent


def ppc_postprocessed(
    waveform, sfreq, phase_lag, coh, fmin, fmax, band_limited=True, random_state=None
):
    coupled = ppc_shifted_copy_with_noise(
        waveform=waveform,
        sfreq=sfreq,
        phase_lag=phase_lag,
        coh=coh,
        fmin=fmin,
        fmax=fmax,
        band_limited=band_limited,
        random_state=random_state,
    )

    b, a = butter(N=2, Wn=np.array([fmin, fmax]) / sfreq * 2, btype="bandpass")
    coupled = filtfilt(b, a, coupled)

    return normalize_variance(coupled)


def simulation_activity_minimal(fwd, info, targets, std, random_state, return_sc=True):
    src = fwd["src"]
    sim = SourceSimulator(src, snr_mode="global")
    sim.add_noise_sources(
        location=select_random, location_params=dict(n=params.n_noise_dipoles)
    )
    sim.add_point_sources(
        location=[(0, t) for t in targets],
        waveform=narrowband_oscillation,
        waveform_params=dict(fmin=params.fmin, fmax=params.fmax),
        std=std,
        names=[str(t) for t in targets],
    )

    sc = sim.simulate(
        sfreq=params.sfreq,
        duration=params.duration,
        random_state=random_state,
        snr_global=params.target_snr,
        snr_params=dict(fmin=params.fmin, fmax=params.fmax),
        fwd=fwd,
    )
    raw = sc.to_raw(fwd, info, sensor_noise_level=params.sensor_noise_level)

    if return_sc:
        return sc, raw

    return raw


def simulation_connectivity_minimal(
    fwd,
    info,
    target_labels,
    interference_labels,
    coupling,
    random_state,
    return_sc=False,
):
    src = fwd["src"]
    extent_mm = area_to_extent(params.conn_example_patch_area_cm2)

    sim = SourceSimulator(src)
    sim.add_noise_sources(
        location=select_random, location_params=dict(n=params.n_noise_dipoles)
    )

    for i_hemi, label in enumerate(target_labels):
        hemi = "rh" if i_hemi else "lh"
        center_vertno = label.center_of_mass(
            subject="fsaverage", restrict_vertices=src, subjects_dir=paths.subjects_dir
        )
        sim.add_patch_sources(
            [(i_hemi, center_vertno)],
            narrowband_oscillation,
            waveform_params=dict(fmin=params.fmin, fmax=params.fmax),
            extents=extent_mm,
            subject="fsaverage",
            subjects_dir=paths.subjects_dir,
            names=[f"target-{hemi}"],
        )

    for i_hemi, label in enumerate(interference_labels):
        hemi = "rh" if i_hemi else "lh"
        center_vertno = label.center_of_mass(
            subject="fsaverage", restrict_vertices=src, subjects_dir=paths.subjects_dir
        )
        sim.add_patch_sources(
            [(i_hemi, center_vertno)],
            narrowband_oscillation,
            waveform_params=dict(fmin=params.fmin, fmax=params.fmax),
            extents=extent_mm,
            subject="fsaverage",
            subjects_dir=paths.subjects_dir,
            names=[f"interference-{hemi}"],
        )

    for edge, edge_params in coupling.items():
        sim.set_coupling(
            edge,
            method=ppc_postprocessed,
            fmin=params.fmin,
            fmax=params.fmax,
            **edge_params,
        )

    sc = sim.simulate(
        sfreq=params.sfreq,
        duration=params.duration,
        fwd=fwd,
        snr_global=params.target_snr,
        snr_params=dict(fmin=params.fmin, fmax=params.fmax),
        random_state=random_state,
    )
    raw = sc.to_raw(fwd, info, sensor_noise_level=params.sensor_noise_level)
    raw.set_eeg_reference(projection=True)

    if return_sc:
        return sc, raw

    return raw


def select_random_in_labels(src, *, labels=None, random_state=None):
    rng = np.random.default_rng(seed=random_state)

    result = []
    for label in labels:
        # Only consider vertices present in the src
        label = label.copy().restrict(src)
        hemi = 0 if label.hemi == "lh" else 1
        vertices = [(hemi, vertno) for vertno in label.vertices]
        selection = rng.choice(vertices, size=1, replace=False)
        result.append(tuple(selection[0]))

    return result


def grow_patches_in_labels(
    src,
    *,
    subject="fsaverage",
    location="random",
    extent=0.0,
    grow_outside=False,
    subjects_dir=None,
    labels=None,
    surf="white",
    random_state=None,
):
    result = []
    for label in labels:
        # NOTE: restrict to ensure that at least one vertex present in the src
        # is selected
        sel_label = label.copy().restrict(src)

        with mne.use_log_level(False):
            patch = mne.label.select_sources(
                subject=subject,
                label=sel_label,
                location=location,
                extent=extent,
                grow_outside=grow_outside,
                subjects_dir=subjects_dir,
                surf=surf,
                random_state=random_state,
            )

        src_idx = 0 if label.hemi == "lh" else 1
        result.append((src_idx, list(patch.vertices)))

    return result


def simulation_activity_reconstruction(
    fwd,
    labels,
    n_noise_dipoles,
    target_snr,
    snr_mode="global",
    std=1,
    source_type="point",
    patch_area_cm2=None,
    n_connections=0,
    mean_coherence=0.0,
    std_coherence=0.0,
    fmin=8,
    fmax=12,
    random_state=None,
):
    assert snr_mode in ["global", "local"]
    assert source_type in ["point", "patch"]
    assert source_type == "point" or patch_area_cm2 is not None

    # Prepare the parameters
    local_snr = None if snr_mode == "global" else target_snr
    source_names = [label.name for label in labels]
    src = fwd["src"]

    sim = SourceSimulator(src, snr_mode=snr_mode)
    sim.add_noise_sources(
        location=select_random, location_params=dict(n=n_noise_dipoles)
    )

    if source_type == "point":
        sim.add_point_sources(
            location=select_random_in_labels,
            location_params=dict(labels=labels),
            waveform=narrowband_oscillation,
            waveform_params=dict(fmin=fmin, fmax=fmax),
            snr=local_snr,
            snr_params=dict(fmin=fmin, fmax=fmax),
            std=std,
            names=source_names,
        )
    else:
        extent_mm = np.sqrt(patch_area_cm2 * 100 / np.pi)

        sim.add_patch_sources(
            location=grow_patches_in_labels,
            location_params=dict(
                labels=labels, extent=extent_mm, subjects_dir=paths.subjects_dir
            ),
            waveform=narrowband_oscillation,
            waveform_params=dict(fmin=fmin, fmax=fmax),
            snr=local_snr,
            snr_params=dict(fmin=fmin, fmax=fmax),
            std=std,
            names=source_names,
        )

    if not n_connections:
        return sim

    # Randomly pick connections that will be active, making sure that
    # no cycles are present
    rng = np.random.default_rng(seed=random_state)
    all_connections = list(combinations(source_names, 2))
    while True:
        logger.info(f"Picking {n_connections} active connections randomly")
        picked_connections = rng.choice(all_connections, n_connections, replace=False)
        graph = nx.Graph()
        graph.add_edges_from(picked_connections)
        if nx.is_forest(graph):
            break

    # Generate random values of coherence and phase lag
    coherence = rng.normal(mean_coherence, std_coherence, (n_connections,))
    coherence = np.clip(coherence, 0.0, 1.0)
    phase_lag = rng.uniform(0.0, 2 * np.pi, (n_connections,))

    # Add the coupling edges
    for (s, e), coh, lag in zip(picked_connections, coherence, phase_lag, strict=True):
        sim.set_coupling(
            (s, e),
            method=ppc_shifted_copy_with_noise,
            phase_lag=lag,
            coh=coh,
            fmin=fmin,
            fmax=fmax,
        )

    return sim


def extract_ground_truth(sc, source_names):
    ground_truth = []
    for name in source_names:
        ground_truth.append(sc[name].waveform)
    stacked = np.vstack(ground_truth)

    logger.info(f"Extracted ground truth activity, shape {stacked.shape}")
    return stacked


def extract_source_cs(sc, fmin, fmax):
    stc = sc.to_stc()

    # Get indices of all vertices with non-zero source activity
    inds = []
    offset = 0
    for idx, s in enumerate(sc.src):
        common, _, ind2 = np.intersect1d(
            stc.vertices[idx], s["vertno"], assume_unique=True, return_indices=True
        )
        assert len(common) == stc.vertices[idx].size

        inds.append(ind2 + offset)
        offset += len(s["vertno"])
    inds = np.hstack(inds)

    # Extract the ground-truth cross-spectra of source activity
    b, a = prepare_filter(sc.sfreq, fmin, fmax)
    data_filt = filtfilt(b, a, stc.data)
    cs_nb = data2cs_hilbert(data_filt)

    rows, cols = np.meshgrid(inds, inds)
    cs_source = csr_matrix(
        (cs_nb.flatten(), (rows.flatten(), cols.flatten())), shape=(offset, offset)
    )

    logger.info(f"Extracted ground-truth cross-spectrum, shape {cs_source.shape}")

    return cs_source


def extract_source_locations(sc, source_names):
    vertno = {}
    for name in source_names:
        source_label = sc[name].to_label(sc.src)
        vertno[name] = source_label.vertices

    logger.info(f"Extracted ground truth locations for {len(vertno)} sources")
    return vertno


def get_source_variance(sc, fwd):
    var = np.vstack(
        [np.var(s.data, axis=-1)[:, np.newaxis] for s in sc._sources.values()]
    )
    vertices = np.vstack([s.vertices for s in sc._sources.values()])

    source_var = np.full((fwd["nsource"],), np.nan)
    offset = 0
    for idx, s in enumerate(fwd["src"]):
        vertices_src = vertices[:, 0] == idx
        sc_vertno = vertices[vertices_src, 1]
        var_vertno = var[vertices_src]
        _, ind1, ind2 = np.intersect1d(sc_vertno, s["vertno"], return_indices=True)
        source_var[offset + ind2] = np.squeeze(var_vertno[ind1])
        offset += len(s["vertno"])

    return source_var
