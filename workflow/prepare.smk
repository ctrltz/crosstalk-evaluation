rule init_workspace:
    localrule: True
    output:
        ".workspace",
        f"{paths.subjects_dir}/.ready"
    shell:
        "python scripts/00_prepare/00_init_workspace.py"

rule prepare_head_model:
    # input:
    #     ".workspace",
    #     f"{paths.subjects_dir}/.ready"
    output:
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-{{spacing}}-fwd.fif"
    shell:
        "python scripts/00_prepare/01_prepare_head_model.py --spacing {wildcards.spacing}"

rule infer_params:
    input:
        # ".workspace",
        # f"{paths.subjects_dir}/.ready",
    output:
        f"{paths.derivatives}/real_data/infer_params/{{subject}}/amplifier_noise.npz",
        expand(
            [
                f"{paths.derivatives}/real_data/infer_params/{{{{subject}}}}/{{condition}}/sensor_space_snr.npz",
                f"{paths.derivatives}/real_data/infer_params/{{{{subject}}}}/{{condition}}/source_space_power_oct6.npz",
                f"{paths.derivatives}/real_data/infer_params/{{{{subject}}}}/{{condition}}/cv_noise_oct6.npz"
            ],
            condition=["EC", "EO"]
        )
    shell:
        "python scripts/00_prepare/02_infer_params_subject.py "
        "--subject {wildcards.subject} "
        "--all"

rule infer_params_ga:
    input:
        # ".workspace",
        # f"{paths.subjects_dir}/.ready"
        expand(
            f"{paths.derivatives}/real_data/infer_params/{{subject}}/amplifier_noise.npz",
            subject=lemon_subject_ids
        ),
        expand(
            [
                f"{paths.derivatives}/real_data/infer_params/{{subject}}/{{condition}}/sensor_space_snr.npz",
                f"{paths.derivatives}/real_data/infer_params/{{subject}}/{{condition}}/source_space_power_oct6.npz",
                f"{paths.derivatives}/real_data/infer_params/{{subject}}/{{condition}}/cv_noise_oct6.npz"
            ],
            condition=["EC", "EO"],
            subject=lemon_subject_ids
        )
    output:
        f"{paths.precomputed}/var_oct6_EC_like.npy",
        f"{paths.precomputed}/var_oct6_EO_like.npy",
        f"{paths.derivatives}/real_data/infer_params/GA/sensor_space_snr.npz",
        f"{paths.derivatives}/real_data/infer_params/GA/amplifier_noise.npz",
        f"{paths.derivatives}/real_data/infer_params/GA/cv_noise_oct6.npz"
    shell:
        "python scripts/00_prepare/03_infer_params_ga.py --normalize-source-power"
        
        
rule prepare_source_depth_ori:
    input:
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif"
    output:
        f"{paths.precomputed}/source_depth.npy",
        f"{paths.precomputed}/source_ori.npy",
    shell: 
        "python scripts/00_prepare/04_source_depth_orientations.py"
