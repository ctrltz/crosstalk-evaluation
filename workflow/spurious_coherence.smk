rule sc_permutations_subject:
    input:
        branch(
            settings.lemon_bids,
            then=expand(
                f"{paths.lemon_data}/{{{{subject}}}}/{{{{subject}}}}_{{condition}}.set",
                condition=["EC", "EO"]
            ),
            otherwise=expand(
                f"{paths.lemon_data}/{{{{subject}}}}_{{condition}}.set",
                condition=["EC", "EO"]
            )
        )
    output:
        expand(
            [
                f"{paths.derivatives}/real_data/spurious_coherence/{{{{subject}}}}/{{condition}}/{{inv_method}}_{{roi_method}}/cs_genuine.npz",
                f"{paths.derivatives}/real_data/spurious_coherence/{{{{subject}}}}/{{condition}}/{{inv_method}}_{{roi_method}}/coh_spurious.npz",
                f"{paths.derivatives}/real_data/spurious_coherence/{{{{subject}}}}/{{condition}}/{{inv_method}}_{{roi_method}}/filters.npz"
            ],
            condition=["EC", "EO"],
            inv_method=['eLORETA'],
            roi_method=['mean', 'mean_flip', 'centroid']
        )
    shell:
        "python scripts/04_spurious_coherence/01_permutations_subject.py "
        "--subject {wildcards.subject}"


rule sc_permutations_ga:
    input:
        expand(
            [
                f"{paths.derivatives}/real_data/spurious_coherence/{{subject}}/{{condition}}/{{inv_method}}_{{roi_method}}/cs_genuine.npz",
                f"{paths.derivatives}/real_data/spurious_coherence/{{subject}}/{{condition}}/{{inv_method}}_{{roi_method}}/coh_spurious.npz",
                f"{paths.derivatives}/real_data/spurious_coherence/{{subject}}/{{condition}}/{{inv_method}}_{{roi_method}}/filters.npz"
            ],
            condition=["EC", "EO"],
            inv_method=['eLORETA'],
            roi_method=['mean', 'mean_flip', 'centroid'],
            subject=lemon_subject_ids
        )
    output:
        f"{paths.derivatives}/real_data/spurious_coherence/grand_average/results_mean.npz"
    shell:
        "python scripts/04_spurious_coherence/02_permutations_ga.py"


rule sc_theory:
    input:
        expand(
            f"{paths.precomputed}/var_oct6_{{condition}}_like.npy",
            condition=["EC", "EO"]
        )
    output:
        expand(
            f"{paths.derivatives}/theory/spurious_coherence/sc_{{condition}}.npz",
            condition=["EC", "EO"]
        )
    shell:
        "python scripts/04_spurious_coherence/03_theory.py"


rule sc_model_comparison:
    input:
        f"{paths.derivatives}/real_data/spurious_coherence/grand_average/results_mean.npz",
        expand(
            f"{paths.derivatives}/theory/spurious_coherence/sc_{{condition}}.npz",
            condition=["EC", "EO"]
        )
    output:
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/distances_DK.npy",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/distances_DK_fit.npy",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/comparison_raw.csv",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/comparison_delta.csv"
    shell:
        "python scripts/04_spurious_coherence/04_comparison.py"
