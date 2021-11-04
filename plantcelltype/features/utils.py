import numpy as np
import math
from numba import njit, types
from numba.typed import Dict
from numba.typed import List


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
    valid_samples = List()
    for i, _point in enumerate(samples):
        d = np.sqrt(np.sum((_point - zeros)**2))
        if d > 1e-5:
            valid_samples.append(_point)
    return valid_samples


@njit()
def check_valid_idx(samples, zeros):
    valid_samples = _check_valid_idx(samples, zeros)
    return valid_samples, len(valid_samples)


def spherical_coo2cartesian(theta, phi, r=None):
    r = np.ones_like(theta) if r is None else r
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def cartesian_coo2spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/(r + 1e-16))
    phi = np.arctan2(y, x)
    return theta, phi, r


def fibonacci_sphere(samples=1):
    """Implementation from as https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere"""
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points
