def construct_matlab_call(wildcards, random_phases):
    return f"rift_save_epochs(" \
           f"'{paths.fieldtrip}', '{paths.rift_scratch}', " \
           f"'{paths.rift}', {wildcards.tag_type}, {random_phases}); exit;"


rule rift_matlab_export_fixed_phases:
    output:
        expand(
            f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_0/preproc/{{subject}}-data_onestim.mat",
            subject=rift_subject_ids
        )
    params:
        matlab_call=lambda wildcards: construct_matlab_call(wildcards, 0)
    shell:
        f'cd matlab && '
        f'{settings.matlab_alias} -nodisplay -nosplash -nodesktop -r '
        f'"{{params.matlab_call}}"'


rule rift_matlab_export_random_phases:
    output:
        expand(
            f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_1/preproc/{{subject}}-data_{{cond}}.mat",
            subject=rift_subject_ids,
            cond=["onestim", "twostim"]
        )
    params:
        matlab_call=lambda wildcards: construct_matlab_call(wildcards, 1)
    shell:
        f'cd matlab && '
        f'{settings.matlab_alias} -nodisplay -nosplash -nodesktop -r '
        f'"{{params.matlab_call}}"'


rule rift_mne_import:
    input:
        expand(
            f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}-data_{{{{cond}}}}.mat",
            subject=rift_subject_ids
        )
    output:
        expand(
            [
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_{{{{cond}}}}-epo.fif",
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_{{{{cond}}}}_stimulation.npy",
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_{{{{cond}}}}_tag1.npy",
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_{{{{cond}}}}_tag2.npy"
            ],
            subject=rift_subject_ids
        ),
        expand(
            f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/info/{{subject}}_{{{{cond}}}}-info.fif",
            subject=rift_subject_ids
        ),
    shell:
        "python scripts/06_rift/01_mne_import.py "
        "--cond {wildcards.cond} "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases}"


rule rift_move_info:
    # Move mne.Info objects from one condition to the root RIFT folder,
    # as Info should not depend on the condition
    input:
        expand(
            f"{paths.rift}/tag_type_1_random_phases_0/info/{{subject}}_onestim-info.fif",
            subject=rift_subject_ids
        )
    output:
        expand(
            f"{paths.rift}/info/{{subject}}-info.fif",
            subject=rift_subject_ids
        )
    shell:
        f'cp -r {paths.rift}/tag_type_1_random_phases_0/info {paths.rift} && '
        f'for i in {paths.rift}/info/*.fif; do mv "$i" "${{{{i/_onestim-info/-info}}}}"; done'


rule rift_headmodels:
    input:
        expand(
            f"{paths.rift}/info/{{subject}}-info.fif",
            subject=rift_subject_ids
        )
    output:
        f"{paths.rift}/ctf_array.npy",
        expand(
            f"{paths.rift}/headmodels/{{subject}}-fwd.fif",
            subject=rift_subject_ids
        ),
        f"{paths.rift}/headmodels/leadfield_mean.npy",
        f"{paths.rift}/info/plot-info.fif"
    shell:
        "python scripts/06_rift/02_prepare_headmodels.py"


rule rift_ssvep_showcase:
    input:
        expand(
            [
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_onestim-epo.fif",
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_onestim_tag1.npy"
            ],
            subject=rift_subject_ids
        ),
        f"{paths.rift}/info/plot-info.fif"
    output:
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/sensor_space/grand_average_onestim_abscoh_stim1.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/roi_space/times.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/roi_space/stim.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/roi_space/v1_left.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/roi_space/v1_left_ga.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/roi_space/v1_right.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/roi_space/v1_right_ga.npy"
    shell:
        "python scripts/06_rift/03_ssvep_showcase.py "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases}"


rule rift_roi_connectivity_subject:
    input:
        f"{paths.rift}/headmodels/{{subject}}-fwd.fif",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/preproc/{{subject}}_data_{{cond}}-epo.fif",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/preproc/{{subject}}_data_{{cond}}_tag1.npy"
    output:
        expand(
            [
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/brain_stimulus/{{{{subject}}}}_{{{{cond}}}}_{{inv_method}}_{{roi_method}}_abscoh_stim1.npy",
                f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/brain_brain/{{{{subject}}}}_{{{{cond}}}}_{{inv_method}}_{{roi_method}}_absimcoh.npy"
            ],
            inv_method=["eLORETA"],
            roi_method=["mean", "mean_flip", "centroid"]
        )
    shell:
        "python scripts/06_rift/04_roi_connectivity_subject.py "
        "--cond {wildcards.cond} "
        "--subject {wildcards.subject} "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases}"


rule rift_roi_connectivity_ga:
    input:
        expand(
            [
                f"{paths.rift}/"
                f"tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/"
                f"brain_stimulus/{{subject}}_{{{{cond}}}}_{{inv_method}}_{{roi_method}}_abscoh_stim1.npy",

                f"{paths.rift}/"
                f"tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/"
                f"brain_brain/{{subject}}_{{{{cond}}}}_{{inv_method}}_{{roi_method}}_absimcoh.npy"
            ],
            subject=rift_subject_ids,
            inv_method=["eLORETA"],
            roi_method=["mean", "mean_flip", "centroid"]
        )
    output:
        f"{paths.rift}/"
        f"tag_type_{{tag_type}}_random_phases_{{random_phases}}/"
        f"brain_stimulus/{{cond}}_avg_abscoh_stim.npy",

        f"{paths.rift}/"
        f"tag_type_{{tag_type}}_random_phases_{{random_phases}}/"
        f"brain_brain/{{cond}}_avg_absimcoh.npy"
    shell:
        "python scripts/06_rift/05_roi_connectivity_ga.py "
        "--cond {wildcards.cond} "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases}"


rule rift_noise_floor:
    input:
        expand(
            f"{paths.rift}/tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/preproc/{{subject}}_data_{{{{cond}}}}-epo.fif",
            subject=rift_subject_ids
        )
    output:
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/theory/noise_floor_{{measure}}_{{cond}}.npz"
    params:
        set_onestim=lambda wildcards: "--onestim" if wildcards.cond == "onestim" else ""
    shell:
        "python scripts/06_rift/06_noise_floor.py "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases} "
        "--measure {wildcards.measure} "
        "{params.set_onestim}"


rule rift_fit_brain_stimulus:
    input:
        f"{paths.rift}/ctf_array.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/theory/noise_floor_coh_onestim.npz",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/brain_stimulus/onestim_avg_abscoh_stim.npy"
    output:
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/brain_stimulus/search_results_{{model}}.csv"
    shell:
        "python scripts/06_rift/07_fit_evaluate.py "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases} "
        "--model {wildcards.model} "
        "--kind brain_stimulus"


rule rift_fit_brain_brain:
    input:
        f"{paths.rift}/ctf_array.npy",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/theory/noise_floor_imcoh_twostim.npz",
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/brain_brain/twostim_avg_absimcoh.npy"
    output:
        f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{random_phases}}/brain_brain/search_results_{{model}}.csv"
    shell:
        "python scripts/06_rift/07_fit_evaluate.py "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases} "
        "--model {wildcards.model} "
        "--kind brain_brain"


rule rift_model_comparison_brain_stimulus:
    input:
        expand(
            f"{paths.rift}/"
            f"tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/"
            f"brain_stimulus/search_results_{{model}}.csv",
            model=["no_leakage", "distance", "ctf"]
        )
    output:
        f"{paths.rift}/"
        f"tag_type_{{tag_type}}_random_phases_{{random_phases}}/"
        f"brain_stimulus/combined_results.csv",

        expand(
            f"{paths.rift}/"
            f"tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/"
            f"brain_stimulus/best_results_{{metric}}_{{option}}_{{target}}.csv",
            metric=["pearson", "spearman"],
            option=["raw", "delta"],
            target=["fit", "all"]
        )
    shell:
        "python scripts/06_rift/08_model_comparison.py "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases} "
        "--measure coh "
        "--kind brain_stimulus "
        "--onestim"


rule rift_model_comparison_brain_brain:
    input:
        expand(
            f"{paths.rift}/"
            f"tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/"
            f"brain_brain/search_results_{{model}}.csv",
            model=["no_leakage", "distance", "ctf"]
        )
    output:
        f"{paths.rift}/"
        f"tag_type_{{tag_type}}_random_phases_{{random_phases}}/"
        f"brain_brain/combined_results.csv",

        expand(
            f"{paths.rift}/"
            f"tag_type_{{{{tag_type}}}}_random_phases_{{{{random_phases}}}}/"
            f"brain_brain/best_results_{{metric}}_{{option}}_{{target}}.csv",
            metric=["pearson", "spearman"],
            option=["raw", "delta"],
            target=["fit", "all"]
        )
    shell:
        "python scripts/06_rift/08_model_comparison.py "
        "--tagging-type {wildcards.tag_type} "
        "--random-phases {wildcards.random_phases} "
        "--measure imcoh "
        "--kind brain_brain"
