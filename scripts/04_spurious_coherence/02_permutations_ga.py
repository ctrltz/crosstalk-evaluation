import numpy as np

from tqdm import tqdm

from ctfeval.config import paths, params, get_pipelines
from ctfeval.datasets import get_lemon_subject_ids
from ctfeval.log import logger
from ctfeval.connectivity import cs2coh, cohy2con, spurious_coherence
from ctfeval.parcellations import PARCELLATIONS


def main():
    # Parcellation
    p = PARCELLATIONS["DK"].load("fsaverage", paths.subjects_dir)

    # Subjects, conditions, pipelines
    subject_ids = get_lemon_subject_ids(params.lemon_bids)
    pipelines = get_pipelines(params.include_data_dependent)
    pipelines = ["_".join(parts) for parts in pipelines]

    n_subjects = len(subject_ids)
    n_conditions = len(params.sc_conditions)
    n_pipelines = len(pipelines)

    sc_path = paths.derivatives / "real_data" / "spurious_coherence"

    # Pick frequencies and dimensions of cross-spectra from any subject
    with np.load(
        sc_path / subject_ids[0] / "EO" / "eLORETA_mean" / "cs_genuine.npz"
    ) as data:
        freqs = data["f"]
        cs = data["cs"]

    # Collect the results from all subjects
    coh_all = np.zeros((n_subjects, n_conditions, n_pipelines, *cs.shape))
    absimcoh_all = np.zeros((n_subjects, n_conditions, n_pipelines, *cs.shape))
    sc_minfreq = np.zeros(
        (n_subjects, n_conditions, n_pipelines, p.n_labels, p.n_labels)
    )
    sc_perm_alpha = np.zeros(
        (n_subjects, n_conditions, n_pipelines, p.n_labels, p.n_labels)
    )

    for i_subject, subject_id in enumerate(tqdm(subject_ids)):
        for i_cond, condition in enumerate(params.sc_conditions):
            for i_pipeline, pipeline in enumerate(pipelines):
                with np.load(
                    sc_path / subject_id / condition / pipeline / "cs_genuine.npz"
                ) as data:
                    try:
                        cohy = cs2coh(data["cs"])
                        coh = cohy2con(cohy, measure="coh")
                        absimcoh = cohy2con(cohy, measure="imcoh", return_abs=True)
                        sc = np.apply_along_axis(spurious_coherence, 0, cohy)

                        coh_all[i_subject, i_cond, i_pipeline, :, :, :] = coh
                        absimcoh_all[i_subject, i_cond, i_pipeline, :, :, :] = absimcoh
                        sc_minfreq[i_subject, i_cond, i_pipeline, :, :] = sc
                    except Exception as e:
                        logger.info(
                            f"[genuine] Failed on {subject_id} - {condition} - {pipeline}: {str(e)}"
                        )

                with np.load(
                    sc_path / subject_id / condition / pipeline / "coh_spurious.npz"
                ) as data:
                    try:
                        coh_spurious = data["coh"]

                        # Average over permutations
                        sc_alpha = coh_spurious.mean(axis=0)

                        sc_perm_alpha[i_subject, i_cond, i_pipeline, :, :] = sc_alpha
                    except Exception as e:
                        logger.info(
                            f"[spurious] Failed on {subject_id} - {condition} - {pipeline}: {str(e)}"
                        )

    # Calculate grand average
    mean_coh = coh_all.mean(axis=0)
    mean_absimcoh = absimcoh_all.mean(axis=0)
    mean_sc_minfreq = sc_minfreq.mean(axis=0)
    mean_sc_perm_alpha = sc_perm_alpha.mean(axis=0)

    # Save the results
    output_path = sc_path / "grand_average"
    output_path.mkdir(exist_ok=True, parents=True)

    np.savez(
        output_path / "results_mean.npz",
        freqs=freqs,
        coh=mean_coh,
        absimcoh=mean_absimcoh,
        sc_minfreq=mean_sc_minfreq,
        sc_perm_alpha=mean_sc_perm_alpha,
    )


if __name__ == "__main__":
    main()
