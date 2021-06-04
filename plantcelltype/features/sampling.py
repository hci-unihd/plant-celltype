from numba import njit
import numpy as np
from numba import types
from numba.typed import Dict


@njit(fastmath=True)
def l2_distance(x0, x1, x2, y):
    return np.sqrt((x0 - y[0]) ** 2 + (x1 - y[1]) ** 2 + (x2 - y[2]) ** 2)


@njit()
def create_mapping_jit(cell_ids):
    cell_mapping = Dict.empty(key_type=types.int64, value_type=types.int64, )
    # create a cell_idx:array_idx
    for i, _ids in enumerate(cell_ids):
        cell_mapping[_ids] = i

    return cell_mapping


def create_mapping(cell_ids):
    cell_mapping = {}
    # create a cell_idx:array_idx
    for i, _ids in enumerate(cell_ids):
        cell_mapping[_ids] = i
    return cell_mapping


@njit(fastmath=True)
def farthest_points_sampling(segmentation, cell_ids, cell_com, n_points=10):
    shape = segmentation.shape
    cell_fps = np.zeros((cell_com.shape[0], n_points + 1, 3))

    # create a cell_idx:array_idx
    cell_mapping = create_mapping_jit(cell_ids)
    # initialize com as first point
    for i, _ids in enumerate(cell_ids):
        cell_fps[i, 0] = cell_com[i]

    # swipe the volume for each point
    for point in range(1, n_points + 1):
        distance_array = np.zeros(cell_ids.shape[0])

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    _label = segmentation[i, j, k]
                    if _label != 0:  # avoid bg
                        _ids = cell_mapping[_label]
                        distance = 0
                        # compute distance to all other points
                        for _point in range(0, point):
                            distance += l2_distance(i, j, k, cell_fps[_ids, _point])

                        # update if distance is the larges
                        if distance > distance_array[_ids]:
                            distance_array[_ids] = distance
                            cell_fps[_ids, point, 0] = i
                            cell_fps[_ids, point, 1] = j
                            cell_fps[_ids, point, 2] = k
    return cell_fps


@njit(fastmath=True)
def _entropy_points_sampling(segmentation, cell_ids, cell_sampling_guess):
    shape = segmentation.shape
    # create a cell_idx:array_idx
    cell_mapping = create_mapping_jit(cell_ids)

    # check for
    start = 0
    for k in range(cell_sampling_guess.shape[1]):
        if cell_sampling_guess[0, k, 0] == 0:
            start = k
            break

    for point in range(start, cell_sampling_guess.shape[1]):
        distance_array = np.zeros(cell_ids.shape[0])
        temp_distance_array = np.zeros(point)
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                for k in range(0, shape[2]):
                    _label = segmentation[i, j, k]
                    if _label != 0:
                        _ids = cell_mapping[_label]
                        for _point in range(0, point):
                            temp_distance_array[_point] = l2_distance(i, j, k, cell_sampling_guess[_ids, _point])

                        entropy = temp_distance_array / np.sum(temp_distance_array)
                        distance = - np.sum(entropy * np.log(entropy + 1e-10))

                        if distance > distance_array[_ids]:
                            distance_array[_ids] = distance
                            cell_sampling_guess[_ids, point, 0] = i
                            cell_sampling_guess[_ids, point, 1] = j
                            cell_sampling_guess[_ids, point, 2] = k
    return cell_sampling_guess


def entropy_points_sampling(segmentation, cell_ids, n_points, n_random_points=10):
    random_points = random_points_samples(segmentation, cell_ids, n_points=n_random_points)
    empty_points = np.zeros((random_points.shape[0], n_points - n_random_points, 3))
    entropy_points = np.concatenate([random_points, empty_points], axis=1)
    return _entropy_points_sampling(segmentation, cell_ids, entropy_points)


def random_points_samples(segmentation, cell_ids, n_points=10):
    segmentation = segmentation.astype('int64')
    cell_ids = cell_ids.astype('int64')
    seg_points = np.nonzero(segmentation)
    random_sampling = np.arange(len(seg_points[0]))
    np.random.shuffle(random_sampling)
    cell_random_sampling = _random_points_samples(seg_points, cell_ids, random_sampling, segmentation, n_points)

    return cell_random_sampling


@njit()
def _random_points_samples(seg_points, cell_ids, random_sampling, segmentation, n_points=10):
    cell_mapping = create_mapping_jit(cell_ids)
    cell_random_sampling = np.zeros((cell_ids.shape[0], n_points, 3), dtype=np.int64)
    counts = np.zeros(cell_ids.shape[0], dtype=np.int64)
    for random_ids in random_sampling:
        i, j, k = seg_points[0][random_ids], seg_points[1][random_ids], seg_points[2][random_ids]
        _idx = cell_mapping[segmentation[i, j, k]]
        if counts[_idx] < n_points:
            cell_random_sampling[_idx, counts[_idx]] = i, j, k
            counts[_idx] += 1

    return cell_random_sampling
