import numpy as np
from numba import njit, prange
from numba.core import types
from numba.typed import Dict

from plantcelltype.utils import create_cell_mapping, create_edge_mapping, check_safe_cast


@njit(parallel=True)
def _mapping2image(in_image, out_image, mappings):
    shape = in_image.shape
    for i in prange(0, shape[0]):
        for j in prange(0, shape[1]):
            for k in prange(0, shape[2]):
                out_image[i, j, k] = mappings[in_image[i, j, k]]

    return out_image


def mapping2image(in_image, mappings, data_type='int64'):
    value_type = types.int64 if data_type == 'int64' else types.float64

    numba_mappings = Dict.empty(key_type=types.int64,
                                value_type=value_type)
    numba_mappings.update(mappings)
    numba_mappings[0] = 0

    out_image = np.zeros_like(in_image).astype(data_type)
    out_image = _mapping2image(in_image, out_image, numba_mappings)
    return out_image


def map_cell_features2segmentation(segmentation, cell_ids, cell_feature):
    out_type = check_safe_cast(cell_feature)
    cell_feature_mapping = create_cell_mapping(cell_ids, cell_feature)
    features_image = mapping2image(segmentation, cell_feature_mapping, out_type)
    return features_image


def map_edges_features2rag_boundaries(rag_boundary, edges_ids, edges_feature):
    out_type = check_safe_cast(edges_feature)
    cell_feature_mapping = create_edge_mapping(edges_ids, edges_feature)
    features_image = mapping2image(rag_boundary, cell_feature_mapping, out_type)
    return features_image
