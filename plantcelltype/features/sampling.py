import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict, List
import torch
from torch_geometric.nn import fps


@njit()
def create_trivial_mapping_jit(cell_ids):
    cell_mapping = Dict.empty(key_type=types.int64, value_type=types.int64, )
    # create a cell_idx:array_idx
    for i, _ids in enumerate(cell_ids):
        cell_mapping[_ids] = i

    return cell_mapping


@njit()
def _random_points_samples(segmentation, cell_ids, seg_points, random_sampling, n_points=10):
    cell_mapping = create_trivial_mapping_jit(cell_ids)
    cell_random_sampling = np.zeros((cell_ids.shape[0], n_points, 3), dtype=np.int64)
    counts = np.zeros(cell_ids.shape[0], dtype=np.int64)
    for random_ids in random_sampling:
        i, j, k = seg_points[0][random_ids], seg_points[1][random_ids], seg_points[2][random_ids]
        _idx = cell_mapping[segmentation[i, j, k]]
        if counts[_idx] < n_points:
            cell_random_sampling[_idx, counts[_idx]] = i, j, k
            counts[_idx] += 1

    return cell_random_sampling


def compute_random_points_samples(segmentation, cell_ids, n_points=10, seed=0):
    segmentation = segmentation.astype('int64')
    cell_ids = cell_ids.astype('int64')
    seg_points = np.nonzero(segmentation)
    random_sampling = np.arange(len(seg_points[0]))
    np.random.seed(seed)
    np.random.shuffle(random_sampling)
    cell_random_sampling = _random_points_samples(segmentation, cell_ids, seg_points, random_sampling, n_points)

    return cell_random_sampling


@njit
def _sort_points_by_cell_type(cell_idx, seg, pos):
    """ sort the position by cell_idx"""
    list_pos = List()

    for query_idx in cell_idx:
        query_pos_list = List()

        for j, idx in enumerate(seg):
            if idx == query_idx:
                query_pos_list.append(pos[j])

        list_pos.append(query_pos_list)
    return list_pos


def compute_farthest_points_samples(segmentation, cell_ids, n_points=10, seed=0, pos_transform=None, min_num_points=3):
    x_seg, y_seg, z_seg = np.nonzero(segmentation)
    idx_seg = segmentation[x_seg, y_seg, z_seg].astype('int64')
    seg_points = np.array([x_seg, y_seg, z_seg]).T

    # transform points
    origin = np.zeros(3)
    if pos_transform is not None:
        seg_points = pos_transform(seg_points)
        origin = pos_transform(origin)

    # sample ids using fps
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sort points by cell_ids
    sorted_by_id_points = _sort_points_by_cell_type(cell_ids, idx_seg, seg_points)

    fps_array = np.zeros((cell_ids.shape[0], n_points, 3))
    for i, id_points in enumerate(sorted_by_id_points):
        # this is required because some samples are empty or too small to be meaningfully sample
        if len(id_points) > min_num_points:
            points_tensor = torch.from_numpy(np.array(id_points)).float()
            points_tensor = points_tensor.to(device)
            idx_samples = fps(points_tensor, ratio=(n_points + 1) / points_tensor.shape[0])

            # :n_points is necessary to get consistent size
            idx_samples = idx_samples[:n_points]
            id_points_samples = points_tensor[idx_samples].cpu().numpy()

            if points_tensor.shape[0] < n_points:
                id_points_samples[points_tensor.shape[0]:, :] = origin

            fps_array[i] = id_points_samples
    return fps_array
