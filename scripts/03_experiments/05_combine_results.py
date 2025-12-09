import pandas as pd

from ctfeval.config import paths, params


def extract_var_mode(simulation_name):
    return simulation_name.partition("var_")[-1].partition("_conn")[0]


def extract_source_size(simulation_name):
    if "point" in simulation_name:
        return "point"

    assert "patch" in simulation_name
    source_size = simulation_name.partition("patch_")[-1].partition("cm2")[0]

    return int(source_size)


def extract_conn_preset(simulation_name):
    return simulation_name.partition("conn_")[-1].partition("_sensor")[0]


def extract_snr(simulation_name):
    return float(simulation_name.partition("snr_")[-1].partition("_var")[0])


def extract_sensor_noise(simulation_name):
    return float(simulation_name.partition("noise_")[-1].partition(".csv")[0])


def main():
    # Paths
    experiments_path = paths.derivatives / "simulations" / "experiments"

    # Load the results
    dfs = []
    for exp in params.experiments:
        corr_df_exp = pd.read_csv(experiments_path / f"exp{exp}_DK_corr_rat.csv")
        corr_df_exp["experiment"] = exp
        dfs.append(corr_df_exp)

    corr_df = pd.concat(dfs)

    corr_df["var_mode"] = corr_df.simulation.apply(extract_var_mode)
    corr_df["source_size"] = corr_df.simulation.apply(extract_source_size)
    corr_df["conn_preset"] = corr_df.simulation.apply(extract_conn_preset)
    corr_df["SNR"] = corr_df.simulation.apply(extract_snr)
    corr_df["sensor_noise"] = corr_df.simulation.apply(extract_sensor_noise)
    corr_df["pipeline"] = corr_df.inv_method + " | " + corr_df.roi_method

    corr_df.to_csv(experiments_path / "combined.csv", index=False)


if __name__ == "__main__":
    main()
