rule theory_ratio_parcellations:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
        f"{paths.precomputed}/source_depth.npy"
    output:
        expand(
            f"{paths.theory}/parcellations/{{parcellation}}.npz",
            parcellation=list(PARCELLATIONS.keys())
        )
    shell:
        "python scripts/02_ctf_ratio/01_ctf_ratio_vs_parcellations.py"


rule theory_ratio_sensors:
    # input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
    output:
        f"{paths.theory}/parcellations/DK_num_channels.csv"
    shell:
        "python scripts/02_ctf_ratio/02_ctf_ratio_vs_sensors.py"


rule theory_imcoh_recovery:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif"
    output:
        f"{paths.theory}/connectivity_estimation/recovery_{{method}}.npy"
    shell:
        "python scripts/05_connectivity_estimation/02_imcoh_recovery.py "
        "--method {wildcards.method}"
