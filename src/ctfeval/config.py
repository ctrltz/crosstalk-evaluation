import numpy as np

from collections import namedtuple
from dotenv import dotenv_values
from itertools import product
from pathlib import Path


env = dotenv_values(".env")

# Paths
asset_path = Path("./assets").absolute()
data_path = Path("./data").absolute()
results_path = Path("./results").absolute()
paths_dict = dict(
    # Project structure
    assets=asset_path,
    data=data_path,
    derivatives=data_path / "derivatives",
    examples=data_path / "derivatives" / "examples",
    figures=results_path / "figures",
    figure_components=results_path / "figures" / "components",
    numbers=results_path / "numbers",
    precomputed=data_path / "derivatives" / "precomputed",
    results=results_path,
    review=data_path / "review",
    rift=data_path / "derivatives" / "real_data" / "rift",
    rift_results=results_path / "real_data" / "rift",
    sanity=results_path / "sanity",
    simulations=data_path / "simulations",
    theory=data_path / "derivatives" / "theory",
    toml=data_path / "derivatives" / "toml",
    # Head model
    head_model_subdir=Path(env["HEAD_MODEL_SUBDIR"]),
    subjects_dir=Path(env["SUBJECTS_DIR"]).expanduser().absolute(),
    # LEMON data
    lemon_data=Path(env["LEMON_DATA"]).expanduser().absolute(),
    lemon_raw_data=Path(env["LEMON_RAW_DATA"]).expanduser().absolute(),
    lemon_behav_data=Path(env["LEMON_BEHAV_DATA"]).expanduser().absolute(),
    # RIFT data
    rift_raw=Path(env["RIFT_RAW"]).expanduser().absolute(),
    rift_scratch=Path(env["RIFT_SCRATCH"]).expanduser().absolute(),
    # Toolboxes
    fieldtrip=Path(env["FIELDTRIP_PATH"]).expanduser().absolute(),
    # Other assets
    lemon_info=asset_path / "lemon-info.fif",
)
PathList = namedtuple("PathList", list(paths_dict.keys()))
paths = PathList(**paths_dict)

# RIFT plotting setup
rift_onestim_params = dict(
    tagging_type=1, random_phases=0, metric="pearson", target="fit"
)
rift_twostim_params = dict(
    tagging_type=1, random_phases=1, metric="pearson", target="fit"
)
RIFTParamsList = namedtuple("RIFTParamsList", rift_onestim_params.keys())
rift_onestim = RIFTParamsList(**rift_onestim_params)
rift_twostim = RIFTParamsList(**rift_twostim_params)

# Params
connectivity_presets = {
    "none": dict(n_connections=0, mean_coherence=0.0, std_coherence=0.0),
    "weak": dict(n_connections=10, mean_coherence=0.25, std_coherence=0.1),
    "strong": dict(n_connections=20, mean_coherence=0.75, std_coherence=0.3),
}
experiments = {
    "1": dict(
        source_area=None,
        snr_mode="global",
        target_snr=3.0,
        var_mode=["equal", "EO_like", "EC_like"],
        conn_preset="none",
        sensor_noise_level=0.01,
    ),
    "2": dict(
        source_area=[None, 2, 4, 8],
        snr_mode="global",
        target_snr=3.0,
        var_mode="EO_like",
        conn_preset="none",
        sensor_noise_level=0.01,
    ),
    "3": dict(
        source_area=2,
        snr_mode="global",
        target_snr=3.0,
        var_mode="EO_like",
        conn_preset=["none", "weak", "strong"],
        sensor_noise_level=0.01,
    ),
    "4a": dict(
        source_area=2,
        snr_mode="global",
        target_snr=[0.33, 1.0, 3.0],
        var_mode="EO_like",
        conn_preset="weak",
        sensor_noise_level=0.01,
    ),
    "4b": dict(
        source_area=2,
        snr_mode="global",
        target_snr=3.0,
        var_mode="EO_like",
        conn_preset="weak",
        sensor_noise_level=[0.01, 0.1, 0.25],
    ),
}
params_dict = dict(
    # Common parameters
    fmin=8,
    fmax=12,
    seed=1234,
    # Simulation parameters (default)
    n_simulations=100,
    sfreq=250,
    duration=120,
    n_noise_dipoles=500,
    connectivity=connectivity_presets,
    snr_mode="global",
    target_snr=3,
    sensor_noise_level=0.01,
    # Factors affecting CTF ratio
    max_ratio_num_channels=[16, 32, 64, 128, 256],
    # Experiments
    expA_unequal_std=3,
    parc_sim="DK",
    spacing_sim="oct6",
    spacing_rec="ico4",
    experiments=experiments,
    theory_same_grid=[
        "source_full_info_auto",
        "source_full_info_actual",
        "roi_full_info_auto",
        "roi_full_info_actual",
        "roi_no_info_auto",
        "roi_no_info_actual",
    ],
    theory_other_grid=[
        "roi_no_info_auto",
        "roi_no_info_actual",
    ],
    # Inverse modeling
    reg=0.05,
    # Spurious coherence
    sc_conditions=["EC", "EO"],
    sc_num_permutations=100,
    sc_fres=1,
    sc_overlap=0.5,
    sc_perm_seg_len=2,
    sc_perm_band=[8, 12],
    # Connectivity example
    conn_example_patch_area_cm2=2,
    conn_example_gt_coh=0.6,
    conn_example_gt_lag=np.pi / 3,
    # RIFT (general)
    rift_fstim=60,
    rift_conditions=["onestim", "twostim"],
    rift_noise_percentile=95,
    rift_noise_simulations=1000,
    rift_lh_search_space="pericalcarine-lh",
    rift_rh_search_space="pericalcarine-rh",
    rift_gammas=[10, 20, 50, 100, 200],
    # RIFT (conditions for plotting)
    rift_onestim=rift_onestim,
    rift_twostim=rift_twostim,
    # Literature review
    review_min_count=5,
    # Inferring from real data
    infer_famp=800,  # frequency used to determine the level of amplifier noise
    infer_tmax=180,  # Length of data used for cross-validation
    infer_n_cv_splits=5,
    infer_reg_min=-4,
    infer_reg_max=1,
    infer_reg_steps=15,
    # Datasets
    lemon_bids=(env["LEMON_BIDS"] == "True"),
    # Pipelines
    include_data_dependent=False,
)
ParamList = namedtuple("ParamList", list(params_dict.keys()))
params = ParamList(**params_dict)

# Settings are used only in Snakemake to avoid name conflict
# with built-in params concept
settings_dict = dict(
    # Datasets
    lemon_bids=params.lemon_bids,
    # Tools
    matlab_alias=env["MATLAB_ALIAS"],
    # Simulations
    n_simulations=params.n_simulations,
    # Experiments
    experiments=list(params.experiments.keys()),
    spacing_sim=params.spacing_sim,
    spacing_rec=params.spacing_rec,
    theory_same_grid=params.theory_same_grid,
    theory_other_grid=params.theory_other_grid,
    # RIFT
    rift_onestim=params.rift_onestim,
    rift_twostim=params.rift_twostim,
)
SettingList = namedtuple("SettingList", list(settings_dict.keys()))
settings = SettingList(**settings_dict)

# Pipelines (data-dependent methods go last so they can be easily excluded)
inv_methods = ["eLORETA", "LCMV"]
roi_methods = ["mean", "mean_flip", "centroid", "pca_flip"]


def get_inverse_methods(include_data_dependent):
    return inv_methods if include_data_dependent else inv_methods[:-1]


def get_roi_methods(include_data_dependent):
    return roi_methods if include_data_dependent else roi_methods[:-1]


def get_pipelines(include_data_dependent, include_reg=False):
    picked_inv_methods = get_inverse_methods(include_data_dependent)
    picked_roi_methods = get_roi_methods(include_data_dependent)

    if not include_reg:
        return list(product(picked_inv_methods, picked_roi_methods))

    return list(product(picked_inv_methods, [params.reg], picked_roi_methods))
