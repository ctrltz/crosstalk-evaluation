import numpy as np
import seaborn as sns


def get_included_papers(df):
    included_pmids = set(df.PMID[df.include == 1])
    return df.copy()[df.PMID.isin(included_pmids)]


def get_counts(df, target_col, group_col="PMID"):
    # Drop duplicates within one paper to avoid inflating the statistics
    # in case one paper tries many different combinations
    df_unique = df[[group_col, target_col]].drop_duplicates()
    return df_unique[target_col].value_counts().reset_index()


def count_edges(df, conditions, group_col="PMID"):
    cols = [group_col]
    for col, value in conditions:
        df = df[df[col] == value]
        cols.append(col)

    df = df[cols].drop_duplicates()

    return len(df)


def prepare_sankey_data(df, considered_values):
    for col in ["modality", "inverse_method", "roi_method"]:
        assert col in considered_values

    # Filter out all rows that have any values beyond the ones that are considered
    modalities = considered_values["modality"]
    inv_methods = considered_values["inverse_method"]
    roi_methods = considered_values["roi_method"]

    df_sel = df.copy()
    all_info_mask = np.logical_and(
        df_sel.modality.isin(modalities),
        np.logical_and(
            df_sel.inverse_method.isin(inv_methods), df_sel.roi_method.isin(roi_methods)
        ),
    )
    df = df_sel[all_info_mask]

    sankey_data = []

    # M/EEG <-> inv
    for meeg in modalities:
        for inv in inv_methods:
            num_edges = count_edges(
                df.copy(), [("modality", meeg), ("inverse_method", inv)]
            )

            if num_edges:
                sankey_data.append([meeg, inv, num_edges])

    # inv <-> ROI
    for inv in inv_methods:
        for roi in roi_methods:
            num_edges = count_edges(
                df.copy(), [("inverse_method", inv), ("roi_method", roi)]
            )

            if num_edges:
                sankey_data.append([inv, roi, num_edges])

    return sankey_data


def plot_paper_counts(
    counts_df,
    target_col,
    ax,
    count_col="count",
    plot_title=None,
    hue_col=None,
    palette=None,
    min_count=None,
    flip=False,
):
    counts_disp = counts_df.copy()
    if min_count is not None:
        keep_mask = counts_disp[count_col] >= min_count
        counts_disp = counts_disp[keep_mask]

    data_cols = dict(x=target_col, y=count_col)
    x_label = ""
    y_label = "Number of papers"
    if flip:
        data_cols = dict(x=count_col, y=target_col)
        x_label = "Number of papers"
        y_label = ""

    sns.barplot(counts_disp, **data_cols, hue=hue_col, palette=palette, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if plot_title is not None:
        ax.set_title(plot_title)

    if hue_col is not None:
        ax.legend(frameon=False, title=None)
