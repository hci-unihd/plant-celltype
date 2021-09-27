import numpy as np
from numba import njit, types
from numba.typed import Dict


@njit()
def label2com(label_image, label_ids):
    shape = label_image.shape
    cell_mapping = Dict.empty(key_type=types.int64, value_type=types.int64)

    for i, _ids in enumerate(label_ids):
        cell_mapping[_ids] = i

    com_vector = np.zeros((label_ids.shape[0], 3))
    counts_vector = np.zeros((label_ids.shape[0]))

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                _label = label_image[i, j, k]
                if _label != 0:
                    _ids = cell_mapping[_label]
                    com_vector[_ids][0] += float(i)
                    com_vector[_ids][1] += float(j)
                    com_vector[_ids][2] += float(k)
                    counts_vector[_ids] += 1

    for i in range(label_ids.shape[0]):
        com_vector[i] /= counts_vector[i]

    return com_vector


def make_seg_hollow(segmentation, rag_boundaries):
    hollow_segmentation = np.zeros_like(segmentation)
    mask = rag_boundaries != 0
    hollow_segmentation[mask] = segmentation[mask]
    return hollow_segmentation


@njit()
def _check_valid_idx(samples, zeros=(0, 0, 0)):
    # pure python version is much slower
    # zeros = np.array(zeros)
    # valid_idx = [i for i, _point in enumerate(samples) if not np.allclose(_point, zeros)]
    valid_idx = []
    for i, _point in enumerate(samples):
        d = np.sqrt(np.sum((_point - zeros)**2))
        if d > 0:
            valid_idx.append(i)
    return valid_idx


def check_valid_idx(samples, zeros=(0, 0, 0)):
    valid_idx = _check_valid_idx(samples, zeros)
    return samples[valid_idx], len(valid_idx)
