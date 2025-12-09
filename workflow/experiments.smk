rule run_experiment:
    input:
        # Combined results of evaluation of baseline methods
        expand(
            f"{paths.derivatives}/simulations/{{simulation_name}}/{{{{parcellation}}}}_{{spacing_rec}}_baseline/combined/{{output_file}}.npy",
            simulation_name=lambda wildcards: get_simulation_names_for_experiment(wildcards.parcellation, settings.spacing_sim, wildcards.experiment),
            spacing_rec=[settings.spacing_sim, settings.spacing_rec],
            output_file=["filters", "pipelines", "results"],
        ),
        # Cross-talk calculations for different methods (simulation and reconstruction grids are the same)
        expand(
            f"{paths.derivatives}/simulations/{{simulation_name}}/{{{{parcellation}}}}_{{spacing_rec}}_baseline/combined/crosstalk_{{theory}}.npy",
            simulation_name=lambda wildcards: get_simulation_names_for_experiment(wildcards.parcellation, settings.spacing_sim, wildcards.experiment),
            spacing_rec=settings.spacing_sim,
            theory=settings.theory_same_grid,
        ),
        # Cross-talk calculations for different methods (different grids for simulation and reconstruction)
        expand(
            f"{paths.derivatives}/simulations/{{simulation_name}}/{{{{parcellation}}}}_{{spacing_rec}}_baseline/combined/crosstalk_{{theory}}.npy",
            simulation_name=lambda wildcards: get_simulation_names_for_experiment(wildcards.parcellation, settings.spacing_sim, wildcards.experiment),
            spacing_rec=settings.spacing_rec,
            theory=settings.theory_other_grid,
        )
    output:
        f"{paths.derivatives}/simulations/experiments/exp{{experiment}}_{{parcellation}}_rec.csv",
        f"{paths.derivatives}/simulations/experiments/exp{{experiment}}_{{parcellation}}_corr_rat.csv"
    shell:
        "python scripts/03_experiments/04_comparison.py "
        "--parcellation {wildcards.parcellation} "
        "--experiment {wildcards.experiment}"


rule combine_experiments:
    input:
        expand(
            f"{paths.derivatives}/simulations/experiments/exp{{experiment}}_DK_corr_rat.csv",
            experiment=settings.experiments
        )
    output:
        f"{paths.derivatives}/simulations/experiments/combined.csv"
    shell:
        "python scripts/03_experiments/05_combine_results.py"
