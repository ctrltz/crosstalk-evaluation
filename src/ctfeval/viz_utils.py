import numpy as np


ZOOM_CONFIG = {"postcentral-lh": dict(focalpoint=np.array([0, 0, 25]), distance=200)}


def crop_screenshot(screenshot):
    """
    Implementation from MNE tutorials is used:
    https://mne.tools/stable/auto_tutorials/visualization/10_publication_figure.html
    """
    # Crop the white space around the brain plot
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

    return cropped_screenshot


def prepare_colormap(data, threshold):
    """
    Default clim: [0, max] for positive data, [-max, max] when negative values
    """
    absmax = np.abs(data).max()
    limits = [0, absmax * threshold, absmax]
    cmap = "Reds"
    if np.any(data < 0):
        limits = [-absmax, 0, absmax]
        cmap = "RdBu_r"

    return limits, cmap


def fill_band_area(ax, freqs, spec, fmin, fmax, color):
    """
    Fill the area between PSD and zero with the specified color.
    """
    band = np.logical_and(freqs >= fmin, freqs <= fmax)
    low = np.zeros(band.sum())
    ax.fill_between(freqs[band], low, spec[band], color=color)


def zoom_in(brain, target):
    param_names = ["roll", "distance", "azimuth", "elevation", "focalpoint"]
    view_dict = dict(zip(param_names, brain.get_view()))
    view_dict.update(ZOOM_CONFIG[target])
    brain.show_view(**view_dict)


def sort_labels(label_names, ratios, return_values=False):
    total = {}
    for name, ratio in zip(label_names, ratios):
        name_no_hemi = name[:-3]
        if name_no_hemi not in total:
            total[name_no_hemi] = 0

        total[name_no_hemi] += ratio

    name_order = sorted(total.items(), key=lambda el: el[1], reverse=True)
    order_lh = [label_names.index(f"{name}-lh") for name, _ in name_order]
    order_rh = [label_names.index(f"{name}-rh") for name, _ in name_order]

    if not return_values:
        return order_lh + order_rh[::-1]

    # NOTE: return average values (across lh and rh, hence * 0.5) that were
    # used for sorting the labels
    values = {f"{name}-lh": value * 0.5 for name, value in name_order}
    values.update({f"{name}-rh": value * 0.5 for name, value in name_order})

    # Sort lexicographically to match the default order of labels in the
    # parcellation, then extract ratio values
    sorted_values = sorted(values.items(), key=lambda el: el[0])
    sorted_values = np.array([el[1] for el in sorted_values])

    return sorted_values


def squeeze_colormap(cmap_original, gray=0.3):
    """
    This function squeezes the whole colormap into half of the [0, 1] range
    so that all colors are used with transparent option of mne.viz.Brain
    """
    cmap_squeezed = cmap_original.resampled(cmap_original.N * 2)
    cmap_squeezed.colors[: cmap_original.N, :3] = gray
    cmap_squeezed.colors[cmap_original.N :, :3] = cmap_original.colors

    return cmap_squeezed
