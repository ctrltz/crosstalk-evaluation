def requires_custom_variance(wildcards):
    return wildcards.var_mode != "equal"


rule simulate_point:
    input:
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-{{spacing}}-fwd.fif",
        branch(
            requires_custom_variance,
            then=f"{paths.precomputed}/var_{{spacing}}_{{var_mode}}.npy"
        )
    output:
        f"{paths.simulations}/"
        f"{{parcellation}}_{{spacing}}_point_"
        f"{{snr_mode}}_snr_{{snr_value}}_"
        f"var_{{var_mode}}_conn_{{conn_preset}}_"
        f"sensor_noise_{{sensor_noise_level}}/simulations.json"
    params:
        n_simulations=settings.n_simulations
    shell:
        "python scripts/03_experiments/00_simulate.py "
        "--spacing {wildcards.spacing} "
        "--parcellation {wildcards.parcellation} "
        "--snr-mode {wildcards.snr_mode} "
        "--target-snr {wildcards.snr_value} "
        "--var-mode {wildcards.var_mode} "
        "--source-type point "
        "--conn-preset {wildcards.conn_preset} "
        "--sensor-noise-level {wildcards.sensor_noise_level} "
        "-n {params.n_simulations} "
        "--overwrite"


rule simulate_patch:
    input:
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-{{spacing}}-fwd.fif",
        branch(
            requires_custom_variance,
            then=f"{paths.precomputed}/var_{{spacing}}_{{var_mode}}.npy"
        )
    output:
        f"{paths.simulations}/"
        f"{{parcellation}}_{{spacing}}_patch_{{area}}cm2_"
        f"{{snr_mode}}_snr_{{snr_value}}_"
        f"var_{{var_mode}}_conn_{{conn_preset}}_"
        f"sensor_noise_{{sensor_noise_level}}/simulations.json"
    params:
        n_simulations=settings.n_simulations
    shell:
        "python scripts/03_experiments/00_simulate.py "
        "--spacing {wildcards.spacing} "
        "--parcellation {wildcards.parcellation} "
        "--snr-mode {wildcards.snr_mode} "
        "--target-snr {wildcards.snr_value} "
        "--var-mode {wildcards.var_mode} "
        "--source-type patch "
        "--patch-area {wildcards.area} "
        "--conn-preset {wildcards.conn_preset} "
        "--sensor-noise-level {wildcards.sensor_noise_level} "
        "-n {params.n_simulations} "
        "--overwrite"


rule evaluate_sim_label:
    input:
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-{{spacing}}-fwd.fif",
        f"{paths.simulations}/{{simulation_name}}/simulations.json"
    output:
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/results_{{label}}.npz"
    shell:
        "python scripts/03_experiments/01_evaluate_sim_label.py "
        "--spacing {wildcards.spacing} "
        "--parcellation {wildcards.parcellation} "
        "--label {wildcards.label} "
        "--simulation {wildcards.simulation_name}"


rule evaluate_sim_combined:
    input:
        lambda wildcards: expand(
            f"{paths.derivatives}/simulations/{{{{simulation_name}}}}/{{{{parcellation}}}}_{{{{spacing}}}}_baseline/results_{{label}}.npz",
            label=lambda wildcards: get_label_names(wildcards.parcellation)
        )
    output:
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/combined/filters.npy",
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/combined/pipelines.npy",
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/combined/results.npy"
    shell:
        "python scripts/03_experiments/02_evaluate_sim_combined.py "
        "--spacing {wildcards.spacing} "
        "--parcellation {wildcards.parcellation} "
        "--simulation {wildcards.simulation_name} "
        "--method {wildcards.parcellation}_{wildcards.spacing}_baseline"


rule evaluate_theory:
    input:
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/combined/filters.npy",
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/combined/pipelines.npy"
    output:
        f"{paths.derivatives}/simulations/{{simulation_name}}/{{parcellation}}_{{spacing}}_baseline/combined/crosstalk_{{mask_mode}}_{{source_cov_mode}}_{{data_cov_mode}}.npy"
    shell:
        "python scripts/03_experiments/03_evaluate_theory.py "
        "--spacing {wildcards.spacing} "
        "--parcellation {wildcards.parcellation} "
        "--simulation {wildcards.simulation_name} "
        "--method {wildcards.parcellation}_{wildcards.spacing}_baseline "
        "--mask {wildcards.mask_mode} "
        "--source-cov {wildcards.source_cov_mode} "
        "--data-cov {wildcards.data_cov_mode}"
