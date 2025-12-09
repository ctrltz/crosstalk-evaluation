rule example_activity:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif"
    output:
        f"{paths.examples}/activity.npz"
    shell:
        "python scripts/01_example_activity/01_minimal_example.py"


rule example_connectivity:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif"
    output:
        f"{paths.examples}/connectivity.npz"
    shell:
        "python scripts/05_connectivity_estimation/01_minimal_example.py"
