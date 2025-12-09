rule numbers_prepare:
    output:
        f"{paths.derivatives}/numbers/durations.npy"
    shell:
        "python scripts/08_numbers/00_prepare.py"


rule numbers_main:
    input:
        f"{paths.lemon_behav_data}/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv",
        f"{paths.derivatives}/numbers/durations.npy",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/comparison_raw.csv",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/comparison_delta.csv",
        
        # RIFT onestim - results for all conditions
        expand(
            f"{paths.rift}/tag_type_{{tag_type}}_random_phases_{{phases}}/brain_stimulus/best_results_{settings.rift_onestim.metric}_{{comparison}}_{settings.rift_onestim.target}.csv",
            tag_type=[1, 4],
            phases=[0, 1],
            comparison=["raw", "delta"]
        ),

        # RIFT twostim - results for all conditions
        expand(
            f"{paths.rift}/tag_type_{{tag_type}}_random_phases_1/brain_brain/best_results_{settings.rift_twostim.metric}_{{comparison}}_{settings.rift_twostim.target}.csv",
            tag_type=[1, 4],
            comparison=["raw", "delta"]
        )
    output:
        f"{paths.numbers}/numbers_main.tex"
    shell:
        "python scripts/08_numbers/01_numbers_main.py"


rule numbers_supplementary:
    input:
        f"{paths.review}/included.csv",
        f"{paths.derivatives}/real_data/infer_params/GA/sensor_space_snr.npz",
        f"{paths.derivatives}/real_data/infer_params/GA/cv_noise_oct6.npz"
    output:
        f"{paths.numbers}/numbers_supplementary.tex"
    shell:
        "python scripts/08_numbers/02_numbers_supplementary.py"
