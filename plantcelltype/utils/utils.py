import argparse
import glob
import math
import os

import numpy as np
from numba import njit


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


def create_features_mapping(features_ids, features, safe_cast=True):
    if safe_cast:
        out_type = check_safe_cast(features)
        features = features.astype(out_type)

    mapping = {}
    for key, value in zip(features_ids, features):
        mapping[key] = value
    return mapping


def create_cell_mapping(cell_ids, cell_feature, safe_cast=True):
    return create_features_mapping(cell_ids, cell_feature, safe_cast=safe_cast)


def create_edge_mapping(edges_ids, edges_features, safe_cast=True):
    cantor_edges_ids = edges_ids2cantor_ids(edges_ids)
    return create_features_mapping(cantor_edges_ids, edges_features, safe_cast=safe_cast)


def check_safe_cast(array, types=('int64', 'float64')):
    for _types in types:
        if np.can_cast(array.dtype, _types):
            out_type = _types
            return out_type
    else:
        raise RuntimeError


def print_config(config, indentation=0):
    spaces = indentation * '  '
    for key, value in config.items():
        if isinstance(value, dict):
            print(f'{spaces}{key}:')
            print_config(value, indentation + 1)
        else:
            print(f'{spaces}{key}: {value}')


def load_paths(path, filter_h5=True):
    if isinstance(path, str) and os.path.isfile(path):
        return [path]

    elif isinstance(path, str):
        files = glob.glob(f'{path}')
        if filter_h5:
            files = list(filter(lambda _path: os.path.splitext(_path)[1] == '.h5', files))
        return files

    elif isinstance(path, list):
        return path


def parser():
    _parser = argparse.ArgumentParser(description='plant-celltype training experiments')
    _parser.add_argument('--config', '-c', type=str, help='Path to the YAML experiments file', required=True)
    args = _parser.parse_args()
    return args
