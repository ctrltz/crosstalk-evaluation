import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import string

from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from ctfeval.config import paths, params
from ctfeval.log import logger
from ctfeval.viz import add_label, set_plot_style

set_plot_style()


def main():
    # Load the results
    experiments_path = paths.derivatives / "simulations" / "experiments"
    corr_df = pd.read_csv(experiments_path / "combined.csv")
    corr_df = corr_df[corr_df.data_cov_mode == "auto"]
    corr_df_no_info = corr_df[
        np.logical_and(
            corr_df.method == f"DK_{params.spacing_rec}_baseline",
            corr_df.mask_mode == "roi",
            corr_df.source_cov_mode == "no_info",
        )
    ]

    # Create the layout
    fig = plt.figure(figsize=(10, 6.5), layout="constrained")
    gs = fig.add_gridspec(
        nrows=5, ncols=5, wspace=0.1, height_ratios=[1, 0.6, 0.2, 1, 0.6]
    )

    # Overview
    metrics_to_plot = np.array(
        [
            "CTF | same | source | full_info",
            "CTF | same | roi | full_info",
            "CTF | another | roi | no_info",
        ]
    )
    hue_metrics = list(
        map(lambda v: "black" if "another" in v else "gray", metrics_to_plot)
    )
    ls_metrics = list(map(lambda v: "--" if "source" in v else "-", metrics_to_plot))

    pipelines_to_plot = np.array(
        [
            "eLORETA | mean",
            "eLORETA | mean_flip",
            "eLORETA | centroid",
            "eLORETA | pca_flip",
            "LCMV | mean",
            "LCMV | mean_flip",
            "LCMV | centroid",
            "LCMV | pca_flip",
        ]
    )
    roi_method_colors = {
        "mean": "#1b9e77",
        "mean_flip": "#d95f02",
        "centroid": "#7570b3",
        "pca_flip": "#e7298a",
    }
    hue_methods = list(
        map(lambda v: roi_method_colors[v.split(" | ")[1]], pipelines_to_plot)
    )
    ls_methods = list(
        map(lambda v: "solid" if "eLORETA" in v else "dashed", pipelines_to_plot)
    )

    plot_grid = zip(
        params.experiments,
        ["var_mode", "source_size", "conn_preset", "SNR", "sensor_noise"],
        [
            "Variance",
            "Source size (cm2)",
            "Connectivity",
            "SNR (dB)",
            "Sensor noise (%)",
        ],
        [
            (["equal", "EO_like", "EC_like"], ["Equal", "EO-like", "EC-like"]),
            (["point", "2", "4", "8"], ["Point", "2", "4", "8"]),
            (["none", "weak", "strong"], ["None", "Weak", "Strong"]),
            (["0.3", "1.0", "3.0"], ["-4.8", "0", "4.8"]),
            (["0.01", "0.1", "0.25"], ["1", "10", "25"]),
        ],
    )
    for i_col, (exp, x_col, x_title, x_values) in enumerate(plot_grid):
        ax_metric = fig.add_subplot(gs[0, i_col])

        x_order, x_labels = x_values
        logger.info(f"{exp=}, {x_col=}, {x_title=}, {x_order=}, {x_labels=}")

        exp_df = corr_df[corr_df.experiment == exp]

        n_levels = len(pd.unique(exp_df[x_col]))
        n_metric_levels = len(pd.unique(exp_df["metric"]))
        nave = len(exp_df) / (n_levels * n_metric_levels)
        logger.info(f"Approximate number of averages (metric): {nave:.1f}")

        sns.pointplot(
            data=exp_df,
            x=x_col,
            y="corr",
            hue="metric",
            ax=ax_metric,
            order=x_order,
            legend=False,
            ms=5,
            lw=1.5,
            hue_order=metrics_to_plot,
            linestyles=ls_metrics,
            palette=hue_metrics,
        )
        ax_metric.set_ylim(0.4, 1.05)

        ax_method = fig.add_subplot(gs[3, i_col])
        exp_df = corr_df_no_info[corr_df_no_info.experiment == exp]

        n_levels = len(pd.unique(exp_df[x_col]))
        n_pipeline_levels = len(pd.unique(exp_df["pipeline"]))
        nave = len(exp_df) / (n_levels * n_pipeline_levels)
        logger.info(f"Approximate number of averages (pipeline): {nave:.1f}")

        sns.pointplot(
            data=exp_df,
            x=x_col,
            y="corr",
            hue="pipeline",
            ax=ax_method,
            order=x_order,
            legend=False,
            ms=5,
            lw=1.5,
            hue_order=pipelines_to_plot,
            linestyles=ls_methods,
            palette=hue_methods,
        )
        ax_method.set_ylim(0.4, 0.8)

        for i_ax, ax in enumerate([ax_metric, ax_method]):
            ax.set_title(f"Experiment {exp}")
            ax.set_xlabel(x_title)
            ax.set_xticks(x_order)
            ax.set_xticklabels(x_labels)
            ax.yaxis.grid(visible=True)

            if i_col:
                ax.spines["left"].set_visible(False)
                ax.set_yticklabels([])
                ax.set_ylabel("")
                for tick in ax.yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
            else:
                add_label(ax, string.ascii_uppercase[i_ax], x=-0.3, y=1.25)
                ax.set_ylabel("Correlation across ROIs")

    # Legends
    sgs_metric = gs[1, 1:4].subgridspec(nrows=1, ncols=2)
    sgs_method = gs[4, 1:4].subgridspec(nrows=1, ncols=2)
    fp_bold = FontProperties(weight="semibold")

    # Source grid
    ax_source_grid = fig.add_subplot(sgs_metric[0])
    legend_elements = []
    descriptions = [
        ("gray", "same grid, covariance is fully known"),
        ("black", "different grid, covariance is not known"),
    ]
    for color, desc in descriptions:
        glyph = Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=desc,
            markerfacecolor=color,
            markersize=5,
        )
        legend_elements.append(glyph)
    ax_source_grid.axis("off")
    ax_source_grid.legend(
        handles=legend_elements,
        loc="center",
        frameon=False,
        title="Source grid and covariance matrix",
        title_fontproperties=fp_bold,
        ncols=1,
        handlelength=2,
        columnspacing=1,
        borderaxespad=0,
    )

    # Mask mode for CTF evaluation
    ax_mask_mode = fig.add_subplot(sgs_metric[1])
    legend_elements = []
    descriptions = [
        ("dashed", "ground-truth source location"),
        ("solid", "ROI containing the source"),
    ]
    for ls, desc in descriptions:
        glyph = Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls,
            label=desc,
        )
        legend_elements.append(glyph)
    ax_mask_mode.axis("off")
    ax_mask_mode.legend(
        handles=legend_elements,
        loc="center",
        frameon=False,
        title="CTF ratio is calculated for",
        title_fontproperties=fp_bold,
        ncols=1,
        handlelength=2,
        columnspacing=1,
        borderaxespad=0,
    )

    # Inverse method
    ax_inv_method = fig.add_subplot(sgs_method[0])
    legend_elements = []
    descriptions = [
        ("solid", "eLORETA"),
        ("dashed", "LCMV"),
    ]
    for ls, desc in descriptions:
        glyph = Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls,
            label=desc,
        )
        legend_elements.append(glyph)
    ax_inv_method.axis("off")
    ax_inv_method.legend(
        handles=legend_elements,
        loc="center",
        frameon=False,
        title="Inverse method",
        title_fontproperties=fp_bold,
        ncols=2,
        handlelength=2,
        columnspacing=1,
        borderaxespad=0,
    )

    ax_roi_method = fig.add_subplot(sgs_method[1])
    legend_elements = []
    descriptions = {
        (c, method.replace("_", "-").replace("pca-flip", "SVD"))
        for method, c in roi_method_colors.items()
    }
    for color, desc in descriptions:
        glyph = Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=desc,
            markerfacecolor=color,
            markersize=5,
        )
        legend_elements.append(glyph)
    ax_roi_method.axis("off")
    ax_roi_method.legend(
        handles=legend_elements,
        loc="center",
        frameon=False,
        title="ROI aggregation method",
        title_fontproperties=fp_bold,
        ncols=2,
        handlelength=2,
        columnspacing=1,
        borderaxespad=0,
    )

    # Save the result
    fig.savefig(paths.figures / "fig5_experiments.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
