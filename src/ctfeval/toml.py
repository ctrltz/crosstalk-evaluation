import numpy as np

from tomlkit import array, nl


def add_array(doc, key, arr, digits=None):
    arr_save = np.squeeze(arr.copy())
    if digits is not None:
        arr_save = np.round(arr_save, digits)

    toml_array = array()
    toml_array.extend(list(arr_save))

    doc.add(nl())
    doc.add(key, toml_array)
    doc.add(nl())
