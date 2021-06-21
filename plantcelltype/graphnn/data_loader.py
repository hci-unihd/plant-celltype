import glob
from plantcelltype.utils import open_full_stack
from plantcelltype.utils.utils import filter_bg_from_edges
from plantcelltype.features.rag import rectify_rag_names
from plantcelltype.utils.axis_transforms import scale_points
import numpy as np

from torch_geometric.data.data import Data
import torch
from skspatial.objects import Vector
from torch_geometric.data import DataLoader


gt_mapping_wb = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 14: 3}


def create_edges_features(stack, axis_transform):
    edges_features = stack['edges_features']
    cell_features = stack['cell_features']
    cell_com_grs = axis_transform.transform_coord(cell_features['cell_com_voxels'])

    edges_com_grs = filter_bg_from_edges(stack['edges_ids'],
                                         axis_transform.transform_coord(edges_features['edges_com_voxels']))
    edges_surface_mu = filter_bg_from_edges(stack['edges_ids'],
                                            axis_transform.scale_volumes(edges_features['edges_surface_voxels']))
    edges_com_distance_mu = filter_bg_from_edges(stack['edges_ids'],
                                                 edges_features['edges_com_distance_mu'])

    edges_ids = rectify_rag_names(stack['cell_ids'], stack['edges_ids'])
    edges_cosine_features = []
    for i, (e1, e2) in enumerate(edges_ids):
        e_v = Vector.from_points(cell_com_grs[e1], cell_com_grs[e2]).unit()

        e1_axis1 = cell_features['cell_lr_axis1_grs'][e1]
        e2_axis1 = cell_features['cell_lr_axis1_grs'][e2]

        e1_axis2 = cell_features['cell_lr_axis2_grs'][e1]
        e2_axis2 = cell_features['cell_lr_axis2_grs'][e2]

        _cosine_features = [np.dot(e1_axis1, e2_axis1),
                            np.dot(e1_axis2, e2_axis2),
                            np.dot(e_v, (e1_axis1 + e2_axis1) / 2.0),
                            np.dot(e_v, (e1_axis2 + e2_axis2) / 2.0)]
        _cosine_features = [abs(_feat) for _feat in _cosine_features]
        edges_cosine_features.append(_cosine_features)

    edges_cosine_features = np.array(edges_cosine_features)
    edges_features_array = np.concatenate([edges_com_grs,
                                           edges_surface_mu[..., None],
                                           edges_com_distance_mu[..., None],
                                           edges_cosine_features], axis=1)
    edges_features_tensors = torch.from_numpy(edges_features_array).float()
    edges_features_tensors = edges_features_tensors - torch.mean(edges_features_tensors, 0)
    edges_features_tensors = edges_features_tensors / torch.std(edges_features_tensors, 0)

    return edges_features_tensors


def create_cell_features(stack, axis_transform):
    cell_features = stack['cell_features']
    cell_com_grs = scale_points(cell_features['cell_com_voxels'], stack['attributes']['element_size_um'])
    cell_volume_mu = axis_transform.scale_volumes(cell_features['cell_volume_voxels'])
    cell_surface_mu = axis_transform.scale_volumes(cell_features['cell_surface_voxels'])
    cell_rw_centrality = cell_features['cell_rw_centrality']
    cell_hops_bg = cell_features['cell_hops_to_bg']
    cell_axis1_grs = axis_transform.inv_transform_coord(cell_features['cell_lr_axis1_grs'], voxel_size=(1, 1, 1))
    cell_axis2_grs = axis_transform.inv_transform_coord(cell_features['cell_lr_axis2_grs'], voxel_size=(1, 1, 1))

    cell_features_array = np.concatenate([cell_axis1_grs,
                                          cell_axis2_grs,
                                          cell_com_grs,
                                          cell_hops_bg[..., None],
                                          cell_rw_centrality[..., None],
                                          cell_volume_mu[..., None],
                                          cell_surface_mu[..., None]], axis=1)

    cell_features_tensors = torch.from_numpy(cell_features_array).float()
    cell_features_tensors = cell_features_tensors - torch.mean(cell_features_tensors, 0)
    cell_features_tensors = cell_features_tensors / torch.std(cell_features_tensors, 0)
    return cell_features_tensors


def create_data(file, sample=100):
    stack, at = open_full_stack(file, keys=['cell_features', 'cell_ids', 'cell_labels', 'edges_features', 'edges_ids'])
    # cell feat
    cell_features_tensors = create_cell_features(stack, at)
    edges_features_tensors = create_edges_features(stack, at)

    new_edges_ids = torch.from_numpy(rectify_rag_names(stack['cell_ids'], stack['edges_ids'])).long()

    labels = stack['cell_labels']
    labels = np.array([gt_mapping_wb[_l] for _l in labels])
    labels = torch.from_numpy(labels.astype('int64')).long()

    cell_type = np.unique(stack['cell_labels'])
    mask_ids = []
    for ct in cell_type:
        ct_mask_ids = np.where(stack['cell_labels'] == ct)[0]
        mask_ids += list(np.random.choice(ct_mask_ids, size=min(sample, ct_mask_ids.shape[0]), replace=False))

    mask = np.zeros_like(stack['cell_ids'])
    mask[mask_ids] = 1
    mask = torch.from_numpy(mask.astype('bool'))

    graph_data = Data(x=cell_features_tensors,
                      y=labels,
                      file_path=file,
                      train_mask=mask,
                      test_mask=~mask,
                      edge_attr=edges_features_tensors,
                      edge_index=new_edges_ids.T)
    return graph_data


def create_loaders(files_list, sample=200, batch_size=1, shuffle=True):
    data = [create_data(file, sample=sample) for file in files_list]

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_random_split(base_path, test_ratio=0.33, seed=0, sample=200, batch_size=1):
    files = glob.glob(base_path)

    np.random.seed(seed)
    np.random.shuffle(files)
    split = int(len(files) * test_ratio)
    files_test, files_train = files[:split], files[split:]

    loader_test = create_loaders(files_test, sample, batch_size, shuffle=False)
    loader_train = create_loaders(files_train, sample, batch_size, shuffle=True)
    return loader_test, loader_train
