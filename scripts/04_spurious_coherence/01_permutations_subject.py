import argparse
import mne
import numpy as np

from numpy.linalg import norm
from tqdm import tqdm

from roiextract.filter import apply_batch_raw

from ctfeval.config import paths, params, get_pipelines
from ctfeval.connectivity import data2cs_fourier, cs2coh, cohy2con
from ctfeval.extraction import (
    apply_inverse_with_weights,
    extract_label_time_course_with_weights,
    get_filter,
)
from ctfeval.log import logger
from ctfeval.io import load_raw, save_filters, init_subject
from ctfeval.permutation import ica_unmix, ica_mix, permute_ica
from ctfeval.prepare import interpolate_missing_channels


parser = argparse.ArgumentParser()
parser.add_argument("--subject", type=str, help="Subject ID", required=True)
parser.add_argument(
    "--parcellation",
    choices=["DK", "Schaefer400"],
    help="The parcellation to use",
    default="DK",
)
parser.add_argument(
    "--permutations",
    type=int,
    help="The number of permutations to perform",
    default=params.sc_num_permutations,
)


def estimate_cross_spectra_baseline(
    raw,
    fwd,
    inv,
    W,
    labels,
    pipeline,
    subjects_dir,
    return_filters=False,
    fres=params.sc_fres,
    overlap=params.sc_overlap,
):
    """
    Parameters
    ----------
    fres: int
        Frequency resolution of the connectivity spectra.
    """
    inv_method, roi_method = pipeline
    src = fwd["src"]

    # Estimate the time course label by label to reduce memory consumption
    label_tc = np.zeros((len(labels), raw.n_times))
    filters = []
    for i, label in enumerate(tqdm(labels, desc="_".join(pipeline))):
        stc = apply_inverse_with_weights(raw, fwd, inv, inv_method, label=label)

        label_tc[i, :], weights = extract_label_time_course_with_weights(
            stc, label, src, roi_method, "fsaverage", subjects_dir, return_weights=True
        )

        sf = get_filter(
            label,
            src,
            W,
            weights,
            inv_method,
            roi_method,
            method_params=dict(reg=params.reg),
            ch_names=fwd["info"]["ch_names"],
        )
        filters.append(sf)

        # Sanity check: time series obtained with MNE methods and with the
        # spatial filter should match, at least up to a scaling constant
        sf_tc = sf.apply_raw(raw)
        dotprod = label_tc[i, :] @ sf_tc.T / (norm(label_tc[i, :]) * norm(sf_tc))
        assert np.allclose(dotprod, 1.0), f"{inv_method}_{roi_method}: {dotprod}"

    # Estimate the cross-spectrum and coherence
    sfreq = raw.info["sfreq"]
    logger.info(f"Estimating the cross-spectra, data shape: {label_tc.shape}")
    f, cs = data2cs_fourier(label_tc, sfreq, fres, overlap)
    logger.info("[done]")

    if not return_filters:
        return f, cs

    return f, cs, filters


def estimate_cross_spectra_filters(
    raw, filters, fres=params.sc_fres, overlap=params.sc_overlap
):
    """
    Parameters
    ----------
    fres: int
        Frequency resolution of the connectivity spectra.
    """

    # Load the filters for each label and estimate label time courses
    logger.info(f"Applying {len(filters)} filters, method={filters[0].method}")
    label_tc = apply_batch_raw(raw, filters)

    # Estimate the cross-spectrum and coherence
    sfreq = raw.info["sfreq"]
    return data2cs_fourier(label_tc, sfreq, fres, overlap)


def main(subject_id, parcellation, n_permutations):
    # Prepare the head model
    fwd, inv, _, p = init_subject(parcellation=parcellation)
    full_info = mne.io.read_info(paths.lemon_info)

    # Output folder
    output_folder = paths.derivatives / "real_data" / "spurious_coherence"
    output_folder.mkdir(parents=True, exist_ok=True)

    # Use all combinations of methods
    baseline_pipelines = get_pipelines(params.include_data_dependent)
    inv_methods = set([p[0] for p in baseline_pipelines])

    # Main loop: estimate spurious coherence on real data
    for cond in params.sc_conditions:
        raw_path = paths.lemon_data / subject_id / f"{subject_id}_{cond}.set"
        if not params.lemon_bids:
            # On the MPI cluster, all files are located in the same folder ;(
            # Temporary fix is to have a separate path, but BIDS in better in the long term
            raw_path = paths.lemon_data / f"{subject_id}_{cond}.set"

        # Load the data and interpolate missing channels
        raw = load_raw(raw_path, interpolate=True, full_info=full_info)

        # Get the full inverse matrix first
        logger.info("Preparing the inverse matrices")
        W = {}
        for inv_method in inv_methods:
            logger.info(f"    {inv_method}")
            _, W[inv_method], _ = apply_inverse_with_weights(
                raw, fwd, inv, inv_method, reg=params.reg, return_matrix=True
            )
        logger.info("[done]")

        # NOTE: Using genuine data, we obtain the spatial filters that correspond
        # to all methods and freeze them when doing permutations, otherwise the
        # data-dependent methods would fit to the permuted data, which is likely wrong
        filters = {}

        # Genuine data, baseline pipelines
        for pipeline in baseline_pipelines:
            pipeline_name = "_".join(pipeline)
            pipeline_folder = output_folder / subject_id / cond / pipeline_name
            pipeline_folder.mkdir(exist_ok=True, parents=True)
            f, cs_genuine, filters[pipeline] = estimate_cross_spectra_baseline(
                raw,
                fwd,
                inv,
                W[inv_method],
                p.labels,
                pipeline,
                paths.subjects_dir,
                return_filters=True,
            )

            cs_path = pipeline_folder / "cs_genuine.npz"
            logger.info(f"Saving genuine cross-spectra to {cs_path}")
            np.savez(cs_path, f=f, cs=cs_genuine)

            filter_path = pipeline_folder / "filters.npz"
            logger.info(f"Saving filters to {filter_path}")
            save_filters(filter_path, filters[pipeline])

        # Sanity check: unmixing and mixing with ICA stored in LEMON works fine
        raw_orig = mne.io.read_raw_eeglab(raw_path, preload=True)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw_orig.set_montage(montage)
        ica_lemon = mne.preprocessing.read_ica_eeglab(raw_path)
        components = ica_unmix(raw_orig, ica_lemon)
        raw_restored = ica_mix(components, ica_lemon)
        assert np.allclose(raw_orig.get_data(), raw_restored)
        logger.info("[sanity] ICA mixing and unmixing is fine")

        # Permute data by shuffling ICA time courses
        # This way, observed coherence should be spurious due to volume conduction
        # Spurious coherence should depend on the leadfield, applied spatial filter
        # and amplitudes of sources -> estimate within a band
        rng = np.random.default_rng(params.seed)
        child_seeds = rng.spawn(n_permutations)
        all_pipelines = baseline_pipelines
        coh_spurious = {
            pipeline: np.zeros((n_permutations, p.n_labels, p.n_labels))
            for pipeline in all_pipelines
        }
        fmin, fmax = params.sc_perm_band
        for i_perm, seed in enumerate(tqdm(child_seeds, desc="permutations")):
            # NOTE: we have to permute first and only then interpolate since
            # ICA does not have the interpolated channels
            raw_perm = permute_ica(
                raw_orig, ica_lemon, seg_len=params.sc_perm_seg_len, random_state=seed
            )
            raw_perm = interpolate_missing_channels(raw_perm, full_info)
            raw_perm.set_eeg_reference(projection=True)

            for pipeline in all_pipelines:
                f, cs_spurious = estimate_cross_spectra_filters(
                    raw_perm, filters[pipeline]
                )
                coh_tmp = cohy2con(cs2coh(cs_spurious), measure="coh")
                band = np.logical_and(f >= fmin, f <= fmax)
                coh_band = coh_tmp[band, :, :].mean(axis=0)
                coh_spurious[pipeline][i_perm, :, :] = coh_band

        for pipeline in all_pipelines:
            pipeline_name = (
                "_".join(pipeline) if isinstance(pipeline, tuple) else pipeline
            )
            pipeline_folder = output_folder / subject_id / cond / pipeline_name

            sc_path = pipeline_folder / "coh_spurious.npz"
            logger.info(f"Saving spurious coherence to {sc_path}")
            np.savez(
                sc_path, band=np.array(params.sc_perm_band), coh=coh_spurious[pipeline]
            )

        logger.info("[done]")


if __name__ == "__main__":
    args = parser.parse_args()
    subject_id = args.subject
    parcellation = args.parcellation
    n_permutations = args.permutations

    logger.info(f"Using {parcellation} parcellation")
    logger.info(f"Processing {subject_id}")
    logger.info(f"Performing {n_permutations} permutations")
    main(subject_id, parcellation, n_permutations)
