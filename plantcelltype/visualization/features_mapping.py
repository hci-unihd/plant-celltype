import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict


@njit(parallel=True)
def _mapping2image(in_image, out_image, mappings):
    shape = in_image.shape
    for i in prange(1, shape[0] - 1):
        for j in prange(1, shape[1] - 1):
            for k in prange(1, shape[2] - 1):
                out_image[i, j, k] = mappings[in_image[i, j, k]]

    return out_image


def mapping2image(in_image, mappings, data_type='int'):
    value_type = types.int64 if data_type == 'int' else types.float64

    numba_mappings = Dict.empty(key_type=types.int64,
                                value_type=value_type)
    numba_mappings.update(mappings)
    numba_mappings[0] = 0

    out_image = np.zeros_like(in_image).astype(data_type)
    out_image = _mapping2image(in_image, out_image, numba_mappings)
    return out_image

