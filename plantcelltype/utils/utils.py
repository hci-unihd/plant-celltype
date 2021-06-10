import math
from numba import njit
import numpy as np


def cantor_sym_depair(z):
    w = math.floor((math.sqrt(8 * z + 1) - 1)/2)
    t = (w**2 + w) / 2
    y = int(z - t)
    x = int(w - y)
    return min(x, y), max(x, y)


@njit
def cantor_sym_pair(k1, k2):
    _k1, _k2 = min(k1, k2), max(k1, k2)
    z = int(0.5 * (_k1 + _k2) * (_k1 + _k2 + 1) + _k2)
    return z


def edges_ids2cantor_ids(edges_ids):
    return np.array([cantor_sym_pair(e1, e2) for e1, e2 in edges_ids])


def filter_bg_from_edges(edges_ids, features, bg=0):
    mask = np.where(np.min(edges_ids, axis=1) != bg)[0]
    return features[mask]


def create_features_mapping(features_ids, features):
    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value
    return mapping


def create_cell_mapping(cell_ids, cell_feature):
    return create_features_mapping(cell_ids, cell_feature)


def create_edge_mapping(edges_ids, edges_features):
    cantor_edges_ids = edges_ids2cantor_ids(edges_ids)
    return create_features_mapping(cantor_edges_ids, edges_features)
