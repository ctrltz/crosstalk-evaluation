import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from ctfeval.config import paths, params
from ctfeval.io import init_subject
from ctfeval.log import logger


def get_best_distance_fit(sc_perm, cond_idx, distances):
    # Expand distances by the number of methods
    # Distance doesn't depend on the method -> all values are the same
    n_methods = sc_perm.shape[1]
    dist_triu = distances[np.triu_indices_from(distances, 1)]
    dist_tiled = np.hstack([dist_triu] * n_methods).reshape(-1, 1)

    # Concatenate values based on real data (triu parts only)
    sc_real_parts = []
    for sc_perm_method in sc_perm[cond_idx, :, :, :]:
        indices = np.triu_indices_from(sc_perm_method, 1)
        sc_real_parts.append(sc_perm_method[indices])

    sc_real = np.hstack(sc_real_parts).reshape(-1, 1)

    # Fit a power-law model:
    #    SC ~ c * dist ^ slope
    #    or
    #    log(SC) ~ log(c) + slope * log(dist)
    fm = LinearRegression().fit(np.log(dist_tiled), np.log(sc_real))
    pred = np.exp(fm.predict(np.log(dist_tiled)))
    intercept = np.exp(fm.intercept_.item(0))
    slope = fm.coef_.item(0)
    cond = params.sc_conditions[cond_idx]
    logger.info(f"Best fit ({cond}): SC = {intercept:.3f} * distance ^ {slope:.3f}")

    # Pick the part for one method
    n_pick = pred.size // n_methods
    pred = np.squeeze(pred[:n_pick])
    distances_fit = np.zeros_like(distances)
    indices = np.triu_indices_from(distances_fit, 1)
    distances_fit[indices] = pred
    distances_fit = distances_fit + distances_fit.T  # make symmetric

    return distances_fit, pred


def main(parcellation="DK"):
    # Load the head model
    fwd, _, _, p = init_subject(parcellation=parcellation)
    src = fwd["src"]

    # Common parameters
    roi_methods = ["mean", "mean_flip", "centroid"]
    n_conditions = len(params.sc_conditions)
    n_methods = len(roi_methods)

    # Output folder
    data_path = paths.derivatives / "real_data" / "spurious_coherence"
    save_path = data_path / "comparison"
    save_path.mkdir(exist_ok=True, parents=True)

    # Load the cross-spectra from permutations
    with np.load(data_path / "grand_average" / "results_mean.npz") as data:
        sc_perm = data["sc_perm_alpha"]

    # Load the theoretical estimates
    sc_theory = {}

    for condition in params.sc_conditions:
        sc_theory[condition] = []

        with np.load(
            paths.derivatives / "theory" / "spurious_coherence" / f"sc_{condition}.npz"
        ) as data:
            for method in roi_methods:
                sc_theory[condition].append(data[method])

        sc_theory[condition] = np.stack(sc_theory[condition], axis=0)
    sc_theory = np.stack(list(sc_theory.values()), axis=0)

    # Calculate distance between all pairs of ROIs
    center_vertno = np.zeros(p.n_labels, dtype=int)
    center_pos = np.zeros((p.n_labels, 3))
    for i_label, label in enumerate(p.labels):
        center_vertno[i_label] = label.center_of_mass(
            subject="fsaverage", restrict_vertices=src, subjects_dir=paths.subjects_dir
        )
        hemi_idx = 0 if label.hemi == "lh" else 1
        center_pos[i_label, :] = src[hemi_idx]["rr"][center_vertno[i_label], :]

    distances = np.sqrt(
        np.sum(
            (center_pos[:, np.newaxis, :] - center_pos[np.newaxis, :, :]) ** 2, axis=2
        )
    )
    np.save(save_path / f"distances_{parcellation}.npy", distances)

    # Evaluate the correspondence between real data and theory
    #   * raw - based on original values of SC
    #   * delta - based on the difference between original values and mean over pipelines
    sc_theory_mean = sc_theory.mean(axis=1)
    sc_perm_mean = sc_perm.mean(axis=1)

    corr_ctf_raw = np.zeros((n_conditions, n_methods))
    corr_ctf_delta = np.zeros((n_conditions, n_methods))
    for i_cond in range(n_conditions):
        for i_method in range(n_methods):
            sc_theory_method = np.squeeze(sc_theory[i_cond, i_method, :, :])
            sc_perm_method = np.squeeze(sc_perm[i_cond, i_method, :, :])
            triu_indices = np.triu_indices_from(sc_theory_method, 1)

            sc_theory_method_delta = sc_theory_method - sc_theory_mean[i_cond, :, :]
            sc_perm_method_delta = sc_perm_method - sc_perm_mean[i_cond, :, :]

            corr_ctf_raw[i_cond, i_method] = pearsonr(
                sc_theory_method[triu_indices], sc_perm_method[triu_indices]
            ).statistic
            corr_ctf_delta[i_cond, i_method] = pearsonr(
                sc_theory_method_delta[triu_indices], sc_perm_method_delta[triu_indices]
            ).statistic

    # Evaluate the correspondence between distance and real data
    corr_dist_raw = np.zeros((n_conditions, n_methods))
    distances_fit = np.zeros((n_conditions, p.n_labels, p.n_labels))
    for i_cond in range(n_conditions):
        distances_fit[i_cond, :, :], pred = get_best_distance_fit(
            sc_perm, i_cond, distances
        )

        for i_method in range(n_methods):
            sc_dist_method = np.squeeze(pred)
            sc_perm_method = np.squeeze(sc_perm[i_cond, i_method, :, :])
            triu_indices = np.triu_indices_from(sc_perm_method, 1)

            corr_dist_raw[i_cond, i_method] = pearsonr(
                sc_dist_method, sc_perm_method[triu_indices]
            ).statistic

    # Pack the results of comparison (raw)
    corr_results = []
    for i_cond, cond in enumerate(params.sc_conditions):
        for i_method in range(n_methods):
            for model, values in zip(
                ["CTF", "Distance"], [corr_ctf_raw, corr_dist_raw]
            ):
                corr_results.append(
                    {
                        "model": model,
                        "condition": cond,
                        "method": roi_methods[i_method],
                        "corr": values[i_cond, i_method],
                        "n": triu_indices[0].size,
                    }
                )
    corr_df = pd.DataFrame(corr_results)

    # Pack the results of comparison (delta)
    corr_delta_results = []
    for i_cond, cond in enumerate(params.sc_conditions):
        for i_method in range(n_methods):
            corr_delta_results.append(
                {
                    "model": "CTF",
                    "condition": cond,
                    "method": roi_methods[i_method],
                    "corr": corr_ctf_delta[i_cond, i_method],
                    "n": triu_indices[0].size,
                }
            )
    corr_delta_df = pd.DataFrame(corr_delta_results)

    # Save the results (distances fit and model comparison)
    np.save(save_path / f"distances_{parcellation}_fit.npy", distances_fit)
    corr_df.to_csv(save_path / "comparison_raw.csv", index=False)
    corr_delta_df.to_csv(save_path / "comparison_delta.csv", index=False)


if __name__ == "__main__":
    main()
