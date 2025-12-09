rule fig1:
    input:
        ".workspace",
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif"
    output:
        f"{paths.figure_components}/fig1_sf.svg",
        f"{paths.figure_components}/fig1_ctf_amplitude.png",
        f"{paths.figure_components}/fig1_ctf_high.png",
        f"{paths.figure_components}/fig1_ctf_low.png",
        f"{paths.figure_components}/fig1_ctf_power.png"
    shell:
        "python scripts/07_figures/fig1_components.py"


rule fig2:
    input:
        ".workspace",
        f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
        f"{paths.precomputed}/var_oct6_EC_like.npy"
    output:
        f"{paths.figure_components}/fig2_sc.png",
        f"{paths.figure_components}/fig2_info.svg",
        f"{paths.figure_components}/fig2_gt.svg",
        f"{paths.figure_components}/fig2_rec.svg",
        f"{paths.figure_components}/fig2_exp1.png",
        f"{paths.figure_components}/fig2_exp2.png",
        f"{paths.figure_components}/fig2_exp3.png",
        f"{paths.figure_components}/fig2_exp4a.svg",
        f"{paths.figure_components}/fig2_exp4b.svg"
    shell:
        "python scripts/07_figures/fig2_components.py"


rule fig3:
    input:
        f"{paths.examples}/activity.npz"
    output:
        f"{paths.figure_components}/fig3_minimal_example.svg",
        f"{paths.figure_components}/fig3_vertex_legend.svg"
    shell:
        "python scripts/07_figures/fig3_minimal_example.py"


rule fig4:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
        expand(
            f"{paths.theory}/parcellations/{{parcellation}}.npz",
            parcellation=list(PARCELLATIONS.keys())
        ),
        f"{paths.theory}/parcellations/DK_num_channels.csv"
    output:
        f"{paths.figures}/fig4_ctf_metrics.png",
        f"{paths.figures}/fig4supp_max_ratios.png"
    shell:
        "python scripts/07_figures/fig4_ctf_ratio.py"


rule fig5:
    input:
        f"{paths.derivatives}/simulations/experiments/combined.csv"
    output:
        f"{paths.figures}/fig5_experiments.png"
    shell:
        "python scripts/07_figures/fig5_experiments.py"


rule fig6:
    input:
        f"{paths.derivatives}/real_data/spurious_coherence/grand_average/results_mean.npz",
        expand(
            f"{paths.derivatives}/theory/spurious_coherence/sc_{{condition}}.npz",
            condition=["EC", "EO"]
        ),
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/distances_DK.npy",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/distances_DK_fit.npy",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/comparison_raw.csv",
        f"{paths.derivatives}/real_data/spurious_coherence/comparison/comparison_delta.csv"
    output:
        f"{paths.figures}/fig6_spurious_coherence.png"
    shell:
        "python scripts/07_figures/fig6_spurious_coherence.py"


rule fig7:
    input:
        f"{paths.examples}/connectivity.npz"
    output:
        f"{paths.figures}/fig7_connectivity_example.png"
    shell:
        "python scripts/07_figures/fig7_connectivity_example.py"


rule fig8:
    input:
        f"{paths.assets}/rift-onestim-setup.png",
        f"{paths.rift}/ctf_array.npy",
        f"{paths.rift}/info/plot-info.fif",
        f"{paths.rift}/headmodels/leadfield_mean.npy",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/sensor_space/grand_average_onestim_abscoh_stim1.npy",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/roi_space/times.npy",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/roi_space/stim.npy",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/roi_space/v1_left_ga.npy",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/theory/noise_floor_coh_onestim.npz",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/brain_stimulus/onestim_avg_abscoh_stim.npy",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/brain_stimulus/best_results_{settings.rift_onestim.metric}_raw_{settings.rift_onestim.target}.csv",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/brain_stimulus/best_results_{settings.rift_onestim.metric}_delta_{settings.rift_onestim.target}.csv"
    output:
        f"{paths.figures}/fig8_rift_onestim.png"
    shell:
        "python scripts/07_figures/fig8_rift_onestim.py"


rule figAsupp:
    input:
        f"{paths.review}/literature_screening.csv",
        f"{paths.review}/literature_manual.csv",
    output:
        f"{paths.figure_components}/figAsupp-barplots.svg",
        f"{paths.figure_components}/figAsupp-sankey.svg",
        f"{paths.review}/included.csv"
    shell:
        "python scripts/07_figures/figAsupp_literature_review.py"


rule figBsupp:
    input:
        f"{paths.theory}/connectivity_estimation/recovery_mean_flip.npy",
        f"{paths.theory}/parcellations/DK.npz"
    output:
        f"{paths.figures}/figBsupp_imcoh_recovery.png"
    shell:
        "python scripts/07_figures/figBsupp_imcoh_recovery.py"


rule figCsupp:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
        f"{paths.derivatives}/real_data/infer_params/GA/sensor_space_snr.npz",
        f"{paths.derivatives}/real_data/infer_params/GA/amplifier_noise.npz",
        f"{paths.derivatives}/real_data/infer_params/GA/cv_noise_oct6.npz",
        f"{paths.precomputed}/var_oct6_EO_like.npy",
        f"{paths.precomputed}/var_oct6_EC_like.npy",
    output:
        f"{paths.figures}/figCsupp_inferred_params.png"
    shell:
        "python scripts/07_figures/figCsupp_inferred_params.py"


rule figDsupp:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
        f"{paths.rift}/ctf_array.npy"
    output:
        f"{paths.figures}/figDsupp_rfs_models.png"
    shell:
        "python scripts/07_figures/figDsupp_rfs_models.py"


rule figEsupp_rift_noise_floor:
    input:
        # f"{paths.subjects_dir}/fsaverage/bem/fsaverage-oct6-fwd.fif",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/theory/noise_floor_coh_onestim.npz",
        f"{paths.rift}/tag_type_{settings.rift_onestim.tagging_type}_random_phases_{settings.rift_onestim.random_phases}/brain_stimulus/onestim_avg_abscoh_stim.npy",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/theory/noise_floor_imcoh_twostim.npz",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/brain_brain/twostim_avg_absimcoh.npy",
    output:
        f"{paths.figures}/figEsupp_rift_noise_floor_onestim.png"
    shell:
        "python scripts/07_figures/figEsupp_rift_noise_floor.py"


rule figEsupp_rift_twostim:
    input:
        f"{paths.assets}/rift-twostim-setup.png",
        f"{paths.rift}/ctf_array.npy",
        f"{paths.rift}/info/plot-info.fif",
        f"{paths.rift}/headmodels/leadfield_mean.npy",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/preproc/sub001_data_twostim-epo.fif",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/preproc/sub001_data_twostim_tag1.npy",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/preproc/sub001_data_twostim_tag2.npy",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/theory/noise_floor_imcoh_twostim.npz",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/brain_brain/twostim_avg_absimcoh.npy",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/brain_brain/best_results_{settings.rift_twostim.metric}_raw_{settings.rift_twostim.target}.csv",
        f"{paths.rift}/tag_type_{settings.rift_twostim.tagging_type}_random_phases_{settings.rift_twostim.random_phases}/brain_brain/best_results_{settings.rift_twostim.metric}_delta_{settings.rift_twostim.target}.csv"
    output:
        f"{paths.figures}/figEsupp_rift_twostim.png"
    shell:
        "python scripts/07_figures/figEsupp_rift_twostim.py"
