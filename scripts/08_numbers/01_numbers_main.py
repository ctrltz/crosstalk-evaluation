import numpy as np
import meegsim
import mne
import pandas as pd
import sys

from ctfeval.config import paths, params
from ctfeval.datasets import get_lemon_subject_ids, get_lemon_age, rift_subfolder
from ctfeval.io import load_source_space
from ctfeval.log import logger
from ctfeval.tex import number2tex, section2tex, string2tex, multirow


def export_environment(f):
    # Environment setup
    logger.info("Environment")
    f.write(section2tex("Environment"))
    f.write(string2tex("versionPython", sys.version.split()[0]))
    f.write(string2tex("versionMNE", mne.__version__))
    f.write(string2tex("versionMEEGsim", meegsim.__version__))


def export_lemon_demographics(f):
    logger.info("LEMON stats and demographics")
    f.write("\n")
    f.write(section2tex("LEMON stats and demographics"))

    # Dmographics
    lemon_subject_ids = get_lemon_subject_ids(False)
    # NOTE: two subjects were skipped in grand averages because of a mismatch
    # in sampling frequency (100 Hz instead of 250 Hz)
    lemon_subject_ids.remove("sub-010276")
    lemon_subject_ids.remove("sub-010277")
    num_included = len(lemon_subject_ids)
    f.write(number2tex("lemonNumIncluded", num_included, digits=0))

    df_demo = pd.read_csv(
        paths.lemon_behav_data
        / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    )
    df_demo = df_demo[df_demo.ID.isin(lemon_subject_ids)]
    num_female = np.sum(df_demo["Gender_ 1=female_2=male"] == 1)
    num_male = np.sum(df_demo["Gender_ 1=female_2=male"] == 2)
    # NOTE: age is only available is 5-year bins
    df_demo.Age = df_demo.Age.apply(get_lemon_age)
    num_young = np.sum(df_demo.Age < 50)
    num_old = np.sum(df_demo.Age > 50)

    f.write(number2tex("lemonNumMale", num_male, digits=0))
    f.write(number2tex("lemonNumFemale", num_female, digits=0))
    f.write(number2tex("lemonNumYoung", num_young, digits=0))
    f.write(number2tex("lemonNumOld", num_old, digits=0))

    # Data length
    conditions = ["EO", "EC"]
    durations = np.load(paths.derivatives / "numbers" / "durations.npy")
    for i_cond, condition in enumerate(conditions):
        avg_duration_minutes = durations[i_cond, :].mean() / 60
        f.write(number2tex(f"lemon{condition}Minutes", avg_duration_minutes, digits=1))


def export_common_parameters(f):
    logger.info("Common parameters")

    src_oct6 = load_source_space("fsaverage", paths.subjects_dir, spacing="oct6")
    src_ico4 = load_source_space("fsaverage", paths.subjects_dir, spacing="ico4")

    f.write("\n")
    f.write(section2tex("Common parameters"))
    f.write(number2tex("fminAlphaHz", params.fmin, digits=0))
    f.write(number2tex("fmaxAlphaHz", params.fmax, digits=0))
    f.write(number2tex("invReg", params.reg, digits=2))
    f.write(number2tex("simOctVertices", src_oct6[0]["nuse"], digits=0))
    f.write(number2tex("simIcoVertices", src_ico4[0]["nuse"], digits=0))


def export_simulation_parameters(f):
    logger.info("Simulation parameters")

    f.write("\n")
    f.write(section2tex("Simulation parameters - Experiments 1-4b"))
    f.write(number2tex("simNumSimulations", params.n_simulations, digits=0))
    f.write(number2tex("simSfreq", params.sfreq, digits=0))
    f.write(number2tex("simDuration", params.duration, digits=0))
    f.write(number2tex("simNumNoiseSources", params.n_noise_dipoles, digits=0))
    f.write(number2tex("simGlobalSNR", params.target_snr, digits=0))
    f.write(number2tex("simGlobalSNRdB", 10 * np.log10(params.target_snr), digits=1))
    f.write(
        number2tex("simSensorNoiseDefault", 100 * params.sensor_noise_level, digits=0)
    )
    f.write(
        number2tex(
            "simPatchSizeDefault", params.experiments["3"]["source_area"], digits=0
        )
    )

    conn_params = [
        ("n_connections", "NumEdges", 0),
        ("mean_coherence", "CohMean", 2),
        ("std_coherence", "CohSD", 1),
    ]
    for conn_preset in ["weak", "strong"]:
        for param, name, digits in conn_params:
            preset = params.connectivity[conn_preset]
            f.write(
                number2tex(
                    f"simConn{conn_preset.title()}{name}", preset[param], digits=digits
                )
            )

    f.write("\n")
    f.write(section2tex("Simulation parameters - Experiments A & C"))
    f.write(number2tex("simExpAUnequalStd", params.expA_unequal_std, digits=0))
    f.write(
        number2tex("connExamplePatchSize", params.conn_example_patch_area_cm2, digits=0)
    )
    f.write(number2tex("connExampleGTCoh", params.conn_example_gt_coh, digits=1))

    lag_to_pi = np.round(np.pi / params.conn_example_gt_lag)
    f.write(string2tex("connExampleGTPhaseLag", f"\\pi/{lag_to_pi:.0f}"))


def export_sc(f):
    logger.info("Spurious coherence")

    # Load results of model comparison, combine raw and delta, pivot over methods
    cmp_path = paths.derivatives / "real_data" / "spurious_coherence" / "comparison"
    df_raw = pd.read_csv(cmp_path / "comparison_raw.csv")
    df_delta = pd.read_csv(cmp_path / "comparison_delta.csv")
    num_analyzed_edges = df_raw.n.unique().item(0)

    df_raw["Comparison"] = "raw"
    df_delta["Comparison"] = "delta"
    for df in [df_raw, df_delta]:
        df.rename(
            columns={"model": "Model", "condition": "Condition", "corr": "Correlation"},
            inplace=True,
        )

    df_combined = pd.concat(
        [
            df.pivot(
                columns=["method"],
                index=["Comparison", "Model", "Condition"],
                values=["Correlation"],
            ).reset_index()
            for df in [df_raw, df_delta]
        ]
    )
    df_combined[("Correlation", "average")] = df_combined["Correlation"].mean(axis=1)
    df_combined["Comparison"] = multirow(
        df_combined["Comparison"].values, skip={"raw": 4, "delta": 2}
    )
    df_combined["Model"] = multirow(
        df_combined["Model"].values, skip={"CTF": 2, "Distance": 2}
    )
    df_combined = df_combined.iloc[:, [0, 1, 2, 4, 5, 3, 6]]

    # Export combined table to TeX, add a horizontal line between raw and delta rows
    latex_table = df_combined.to_latex(
        index=False,
        column_format="ccccccc",
        multicolumn_format="c",
        float_format="%.2f",
    ).replace("mean_flip", "mean-flip")
    before, at, after = latex_table.partition("\\multirow{2}{*}{delta}")
    latex_table = before + "\\midrule\n" + at + after

    # Explained variance between pipelines
    delta_ev = np.mean(df_combined.iloc[-2:, 3:6].values ** 2)

    f.write("\n")
    f.write(section2tex("Spurious coherence"))
    f.write(number2tex("scNumPermutations", params.sc_num_permutations, digits=0))
    f.write(number2tex("scPermSegLen", params.sc_perm_seg_len, digits=0))
    f.write(number2tex("scNumDKConnections", num_analyzed_edges, digits=0))
    f.write(string2tex("scResultsTable", latex_table))
    f.write(number2tex("scCTFDeltaEV", delta_ev * 100, digits=0))


def export_rift(f):
    logger.info("RIFT")

    # Onestim - collect and export the results
    metric_raw = f"{params.rift_onestim.metric}_raw_{params.rift_onestim.target}"
    metric_delta = f"{params.rift_onestim.metric}_delta_{params.rift_onestim.target}"

    results_raw = []
    results_delta = []
    for tagging_type in [1, 4]:
        for random_phases in [0, 1]:
            tag_folder = paths.rift / rift_subfolder(tagging_type, random_phases)
            brain_stimulus_path = tag_folder / "brain_stimulus"

            df_raw = pd.read_csv(brain_stimulus_path / f"best_results_{metric_raw}.csv")
            df_raw = df_raw.loc[:, ["model", metric_raw]]
            df_raw.rename(
                columns={"model": "Model", metric_raw: "Correlation"}, inplace=True
            )
            df_raw["Tagging type"] = tagging_type
            df_raw["Stimulus phase"] = "random" if random_phases else "fixed"
            df_raw["Comparison"] = "raw"
            results_raw.append(df_raw)

            df_delta = pd.read_csv(
                brain_stimulus_path / f"best_results_{metric_delta}.csv"
            )
            df_delta = df_delta.loc[:, ["model", metric_delta]]
            df_delta.rename(
                columns={"model": "Model", metric_delta: "Correlation"}, inplace=True
            )
            df_delta["Tagging type"] = tagging_type
            df_delta["Stimulus phase"] = "random" if random_phases else "fixed"
            df_delta["Comparison"] = "delta"
            results_delta.append(df_delta)

    df_onestim = (
        pd.concat(results_raw + results_delta)
        .pivot(
            columns=["Model"],
            index=["Comparison", "Tagging type", "Stimulus phase"],
            values=["Correlation"],
        )
        .reset_index()
        .sort_values(
            by=["Comparison", "Tagging type", "Stimulus phase"],
            ascending=[False, True, True],
        )
    )

    df_onestim["Comparison"] = multirow(
        df_onestim["Comparison"].values, skip={"raw": 4, "delta": 4}
    )
    df_onestim["Tagging type"] = multirow(
        df_onestim["Tagging type"].values, skip={1: 2, 4: 2}
    )
    df_onestim = df_onestim.iloc[:, [0, 1, 2, 5, 4, 3]]

    onestim_table = (
        df_onestim.to_latex(
            index=False,
            column_format="cccccc",
            multicolumn_format="c",
            float_format="%.2f",
        )
        .replace("no_leakage", "No RFS")
        .replace("distance", "Distance")
        .replace("ctf", "CTF")
        .replace("NaN", "---")
    )
    before, at, after = onestim_table.partition("\\multirow{4}{*}{delta}")
    onestim_table = before + "\\midrule\n" + at + after

    onestim_delta_ev = 100 * df_onestim.iloc[4:, -1].values ** 2

    # Twostim - collect and export the results
    metric_raw = f"{params.rift_twostim.metric}_raw_{params.rift_twostim.target}"
    metric_delta = f"{params.rift_twostim.metric}_delta_{params.rift_twostim.target}"

    results_raw = []
    results_delta = []
    for tagging_type in [1, 4]:
        # NOTE: two-stimuli condition was presented only with random phases
        tag_folder = paths.rift / rift_subfolder(tagging_type, 1)
        brain_brain_path = tag_folder / "brain_brain"

        df_raw = pd.read_csv(brain_brain_path / f"best_results_{metric_raw}.csv")
        df_raw = df_raw.loc[:, ["model", metric_raw]]
        df_raw.rename(
            columns={"model": "Model", metric_raw: "Correlation"}, inplace=True
        )
        df_raw["Tagging type"] = tagging_type
        df_raw["Comparison"] = "raw"
        results_raw.append(df_raw)

        df_delta = pd.read_csv(brain_brain_path / f"best_results_{metric_delta}.csv")
        df_delta = df_delta.loc[:, ["model", metric_delta]]
        df_delta.rename(
            columns={"model": "Model", metric_delta: "Correlation"}, inplace=True
        )
        df_delta["Tagging type"] = tagging_type
        df_delta["Comparison"] = "delta"
        results_delta.append(df_delta)

    df_twostim = (
        pd.concat(results_raw + results_delta)
        .pivot(
            columns=["Model"],
            index=["Comparison", "Tagging type"],
            values=["Correlation"],
        )
        .reset_index()
        .sort_values(
            by=["Comparison", "Tagging type"],
            ascending=[False, True],
        )
    )

    df_twostim["Comparison"] = multirow(
        df_twostim["Comparison"].values, skip={"raw": 2, "delta": 2}
    )
    df_twostim = df_twostim.iloc[:, [0, 1, 4, 3, 2]]
    twostim_delta_ev = 100 * df_twostim.iloc[2:, -1].values ** 2

    twostim_table = (
        df_twostim.to_latex(
            index=False,
            column_format="cccccc",
            multicolumn_format="c",
            float_format="%.2f",
        )
        .replace("no_leakage", "No RFS")
        .replace("distance", "Distance")
        .replace("ctf", "CTF")
        .replace("NaN", "---")
    )
    before, at, after = twostim_table.partition("\\multirow{2}{*}{delta}")
    twostim_table = before + "\\midrule\n" + at + after

    f.write("\n")
    f.write(section2tex("RIFT"))
    f.write(number2tex("riftStimFreq", params.rift_fstim, digits=0))
    f.write(
        number2tex("riftNoiseFloorPercentile", params.rift_noise_percentile, digits=0)
    )
    f.write(
        number2tex(
            "riftNoiseFloorNumSimulations", params.rift_noise_simulations, digits=0
        )
    )
    f.write(string2tex("riftOnestimResults", onestim_table))
    f.write(number2tex("riftOnestimDeltaEVMin", onestim_delta_ev.min(), digits=0))
    f.write(number2tex("riftOnestimDeltaEVMax", onestim_delta_ev.max(), digits=0))
    f.write(string2tex("riftTwostimResults", twostim_table))
    f.write(number2tex("riftTwostimDeltaEVMin", twostim_delta_ev.min(), digits=0))
    f.write(number2tex("riftTwostimDeltaEVMax", twostim_delta_ev.max(), digits=0))


def main():
    f = open(paths.numbers / "numbers_main.tex", "w+")
    try:
        export_environment(f)
        export_lemon_demographics(f)
        export_common_parameters(f)
        export_simulation_parameters(f)
        export_sc(f)
        export_rift(f)
    finally:
        f.close()


if __name__ == "__main__":
    main()
