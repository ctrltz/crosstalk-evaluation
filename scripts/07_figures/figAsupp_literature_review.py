import holoviews as hv
import matplotlib.pyplot as plt
import pandas as pd

from ctfeval.config import paths, params
from ctfeval.log import logger
from ctfeval.review import (
    get_included_papers,
    get_counts,
    plot_paper_counts,
    prepare_sankey_data,
)
from ctfeval.viz import add_label, set_plot_style

hv.extension("matplotlib")
set_plot_style()

INV_FAMILY = {
    "LCMV": "Beamformers",
    "SAM": "Beamformers",
    "sLORETA": "Minimum-norm",
    "eLORETA": "Minimum-norm",
    "wMNE": "Minimum-norm",
    "dSPM": "Minimum-norm",
    "MNE": "Minimum-norm",
    "LAURA": "Minimum-norm",
}
INV_PALETTE = {
    "Beamformers": "tab:orange",
    "Minimum-norm": "tab:blue",
    "Other": "tab:gray",
}
PARC_TYPE = {
    "DK": "Anatomical",
    "AAL": "Anatomical",
    "BA": "Cytoarchitectonic",
    "Destrieux": "Anatomical",
    "HCP": "Multimodal",
    "DKT": "Anatomical",
    "Schaefer": "Functional",
}
PARC_PALETTE = {
    "Anatomical": "#a6cee3",
    "Cytoarchitectonic": "#b2df8a",
    "Multimodal": "#1f78b4",
    "Functional": "#33a02c",
    "Other": "tab:gray",
}


def font_size_hook(plot, element):
    for el in plot.handles["labels"]:
        el.set_fontsize("x-large")
        el.set_fontname("Arial")


def main():
    # Load the extracted information
    df_screening = pd.read_csv(paths.review / "literature_screening.csv", header=1)
    df_manual = pd.read_csv(paths.review / "literature_manual.csv", header=0)
    df_included = pd.concat(
        [get_included_papers(df_screening), get_included_papers(df_manual)],
        ignore_index=True,
    )
    df_included.to_csv(f"{paths.review}/included.csv")

    # Set up the figure
    fig = plt.figure(figsize=(10, 3), layout="constrained")
    g = fig.add_gridspec(nrows=1, ncols=3)

    # Inverse method
    logger.info("Histogram for inverse methods")
    ax_inv = fig.add_subplot(g[0, 0])
    inv_df = get_counts(df_included, "inverse_method")
    inv_df["family"] = inv_df.inverse_method.apply(lambda x: INV_FAMILY.get(x, "Other"))
    plot_paper_counts(
        inv_df,
        "inverse_method",
        ax_inv,
        hue_col="family",
        palette=INV_PALETTE,
        min_count=params.review_min_count,
        plot_title="Inverse method",
        flip=True,
    )
    add_label(ax_inv, "A")

    # ROI aggregation
    logger.info("Histogram for ROI aggregation methods")
    ax_roi = fig.add_subplot(g[0, 1])
    roi_df = get_counts(df_included, "roi_method")
    plot_paper_counts(
        roi_df,
        "roi_method",
        ax_roi,
        min_count=params.review_min_count,
        plot_title="Aggregation method",
        flip=True,
    )
    add_label(ax_roi, "B")

    # Parcellations
    logger.info("Histogram for parcellations")
    ax_atlas = fig.add_subplot(g[0, 2])
    atlas_df = get_counts(df_included, "atlas")
    atlas_df["type"] = atlas_df.atlas.apply(lambda x: PARC_TYPE.get(x, "Other"))
    plot_paper_counts(
        atlas_df,
        "atlas",
        ax_atlas,
        hue_col="type",
        palette=PARC_PALETTE,
        min_count=params.review_min_count,
        plot_title="Parcellation",
        flip=True,
    )
    add_label(ax_atlas, "C")

    # Save the result
    fig.savefig(
        paths.figure_components / "figAsupp-barplots.svg",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # Pipelines
    logger.info("Collecting pipeline data")
    sankey_data = prepare_sankey_data(
        df_included,
        {
            "modality": ["MEG", "EEG"],
            "inverse_method": [
                "sLORETA",
                "eLORETA",
                "LCMV",
                "wMNE",
                "dSPM",
                "MNE",
                "LAURA",
                "SAM",
            ],
            "roi_method": ["mean", "mean_flip", "centroid", "SVD"],
        },
    )
    sankey_df = pd.DataFrame(sankey_data, columns=["source", "target", "value"])

    logger.info("Sankey diagram for pipelines")
    pipelines = hv.Sankey(sankey_df)
    pipelines.opts(hooks=[font_size_hook])
    hv.save(
        pipelines, paths.figure_components / "figAsupp-sankey.svg", backend="matplotlib"
    )


if __name__ == "__main__":
    main()
