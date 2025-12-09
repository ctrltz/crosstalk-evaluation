import mne
import numpy as np

from itertools import product

from roiextract.utils import get_label_mask

from ctfeval.config import paths, params
from ctfeval.parcellations import PARCELLATIONS
from ctfeval.io import get_simulation_name


def labeldata2data(label_data, labels, src):
    n_voxels = sum(len(s["vertno"]) for s in src)
    data = np.zeros((n_voxels,))
    is_dict = isinstance(label_data, dict)

    len_provided = len(label_data) if is_dict else label_data.size
    assert len_provided == len(labels), "Expected one value per label"

    for i, label in enumerate(labels):
        mask = get_label_mask(label, src)
        if is_dict:
            data[mask] = label_data[label.name]
        else:
            data[mask] = label_data[i]

    return data


def data2stc(data, src):
    vertno = [s["vertno"] for s in src]
    n_voxels = sum(len(el) for el in vertno)
    assert data.size == n_voxels, "Expected one value per source"
    return mne.SourceEstimate(
        data=data, vertices=vertno, tmin=0, tstep=0.01, subject="fsaverage"
    )


def unpack_experiment_params(exp):
    param_values = []
    for name, values in params.experiments[exp].items():
        if isinstance(values, list):
            param_values.append([(name, v) for v in values])
            continue

        param_values.append([(name, values)])

    return param_values


def get_simulation_names_for_experiment(parc_code, spacing, exp):
    param_values = [
        [("parc_code", parc_code)],
        [("spacing", spacing)],
        [("name_extra", "")],
    ]
    param_values.extend(unpack_experiment_params(exp))
    combinations = list(product(*param_values))

    return [get_simulation_name(**dict(comb)) for comb in combinations]


def get_label_names(code):
    p = PARCELLATIONS[code].load("fsaverage", paths.subjects_dir)
    return p.label_names


def vertno_to_index(src, hemi, vertno):
    assert hemi in ["lh", "rh"]
    if hemi == "lh":
        return np.where(src[0]["vertno"] == int(vertno))[0][0]
    else:
        return src[0]["nuse"] + np.where(src[1]["vertno"] == int(vertno))[0][0]


def area_to_extent(area_cm2):
    return np.sqrt(area_cm2 * 100 / np.pi)
