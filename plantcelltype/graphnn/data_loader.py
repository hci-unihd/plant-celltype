import glob
import os

import numpy as np
import torch
from skspatial.objects import Vector
from torch_geometric.data import DataLoader
from torch_geometric.data.data import Data
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from plantcelltype.features.norms import quantile_zscore, feat_to_bg_onehot, quantile_norm
from plantcelltype.features.norms import quantile_robust_zscore
from plantcelltype.features.rag import rectify_rag_names
from plantcelltype.utils import open_full_stack
from plantcelltype.utils.utils import filter_bg_from_edges

gt_mapping_wb = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 14: 3}


def collect_edges_features(stack, axis_transform):
    edges_features = stack['edges_features']
    cell_features = stack['cell_features']
    cell_com_grs = axis_transform.transform_coord(cell_features['com_voxels'])

    edges_com_grs = filter_bg_from_edges(stack['edges_ids'],
                                         axis_transform.transform_coord(edges_features['com_voxels']))
    edges_surface_mu = filter_bg_from_edges(stack['edges_ids'],
                                            axis_transform.scale_volumes(edges_features['surface_voxels']))
    edges_com_distance_mu = filter_bg_from_edges(stack['edges_ids'],
                                                 edges_features['com_distance_um'])

    edges_ids = rectify_rag_names(stack['cell_ids'], stack['edges_ids'])
    edges_cosine_features = []
    for i, (e1, e2) in enumerate(edges_ids):
        e_v = Vector.from_points(cell_com_grs[e1], cell_com_grs[e2]).unit()

        e1_axis1 = cell_features['lr_axis1_grs'][e1]
        e2_axis1 = cell_features['lr_axis1_grs'][e2]

        e1_axis2 = cell_features['lr_axis2_grs'][e1]
        e2_axis2 = cell_features['lr_axis2_grs'][e2]

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


def collect_cell_features_grs(stack, axis_transform, as_array=True):
    cell_features = stack['cell_features']

    list_feat = [quantile_zscore(axis_transform.transform_coord(cell_features['com_voxels'])),
                 quantile_zscore(axis_transform.scale_volumes(cell_features['volume_voxels'])),
                 quantile_zscore(axis_transform.scale_volumes(cell_features['surface_voxels'])),
                 feat_to_bg_onehot(cell_features['hops_to_bg'], max_channel=5, extreme=(-1, 1))
                 ]

    for zscore_feat in ['length_axis1_grs',
                        'length_axis2_grs',
                        'length_axis3_grs',
                        'pca_explained_variance_grs',
                        'com_proj_grs',
                        'rw_centrality',
                        'degree_centrality',
                        ]:
        list_feat.append(quantile_zscore(cell_features[zscore_feat]))

    for dot_feat in ['lrs_proj_axis1_grs',
                     'lrs_proj_axis2_grs',
                     'lrs_proj_axis3_grs',
                     'pca_proj_axis1_grs',
                     'pca_proj_axis2_grs',
                     'pca_proj_axis3_grs',
                     'lr_axis1_grs',
                     'lr_axis2_grs',
                     'lr_axis3_grs',
                     'pca_axis1_grs',
                     'pca_axis2_grs',
                     'pca_axis3_grs'
                     ]:
        list_feat.append(quantile_zscore(cell_features[dot_feat]))

    list_feat = [feat if feat.ndim == 2 else feat[:, None] for feat in list_feat]
    list_feat = np.concatenate(list_feat, axis=1) if as_array else list_feat
    return list_feat


def create_data(file):
    stack, at = open_full_stack(file, keys=['cell_features',
                                            'cell_ids',
                                            'cell_labels',
                                            'edges_features',
                                            'edges_ids',
                                            'edges_labels'])

    # cell feat
    cell_features_tensors = torch.from_numpy(collect_cell_features_grs(stack, at)).float()
    edges_features_tensors = None  # torch.from_numpy(collect_edges_features(stack, at)).float()
    new_edges_ids = torch.from_numpy(rectify_rag_names(stack['cell_ids'], stack['edges_ids'])).long()

    # create labels
    labels = stack['cell_labels']
    labels = np.array([gt_mapping_wb[_l] for _l in labels])
    labels = torch.from_numpy(labels.astype('int64')).long()

    edges_labels = stack['edges_labels']
    edges_labels = filter_bg_from_edges(stack['edges_ids'], edges_labels)
    edges_labels = torch.from_numpy(edges_labels.astype('int64')).long()

    stage = stack['attributes']['stage']
    pos = torch.from_numpy(stack['cell_features']['com_voxels'])

    # build torch_geometric Data obj
    graph_data = Data(x=cell_features_tensors,
                      y=labels,
                      pos=pos,
                      file_path=file,
                      stage=stage,
                      edge_attr=edges_features_tensors,
                      edge_y=edges_labels,
                      edge_index=new_edges_ids.T)
    return graph_data


def create_loaders(files_list, batch_size=1, shuffle=True):
    data = [create_data(file) for file in files_list]

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_random_split(base_path, test_ratio=0.33, seed=0):
    files = glob.glob(base_path)

    np.random.seed(seed)
    np.random.shuffle(files)
    split = int(len(files) * test_ratio)
    files_test, files_train = files[:split], files[split:]
    return files_test, files_train


def get_stage_random_split(base_path, test_ratio=0.33, seed=0):
    files = glob.glob(base_path)

    all_stages = np.unique([os.path.split(file)[0] for file in files])
    files_test, files_train = [], []
    for stage in all_stages:
        stage_files = sorted(glob.glob(f'{stage}/*.h5'))
        np.random.seed(seed)
        np.random.shuffle(stage_files)
        split = int(len(stage_files) * test_ratio)
        files_test += stage_files[:split]
        files_train += stage_files[split:]

    return files_test, files_train


class ConvertGeometricDataSet(TorchDataset):
    def __init__(self, geometric_loader):
        list_x, list_y = [], []
        for data in geometric_loader:
            list_x.append(data.x)
            list_y.append(data.y)

        self.x = torch.cat(list_x, 0)
        self.y = torch.cat(list_y, 0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


def build_geometric_loaders(base_path, test_ratio=0.33, seed=0, batch_size=1, mode='stage_random'):
    if mode == 'stage_random':
        files_test, files_train = get_stage_random_split(base_path, test_ratio=test_ratio, seed=seed)
    elif mode == 'random':
        files_test, files_train = get_random_split(base_path, test_ratio=test_ratio, seed=seed)
    else:
        raise NotImplemented

    loader_test = create_loaders(files_test, batch_size, shuffle=False)
    loader_train = create_loaders(files_train, batch_size, shuffle=True)
    num_feat = loader_train.dataset[0].x.shape[-1]
    return loader_test, loader_train, num_feat


def build_standard_loaders(base_path, test_ratio=0.33, seed=0, batch_size=1, mode='stage_random'):
    loader_g_test, loader_g_train, num_feat = build_geometric_loaders(base_path,
                                                                      test_ratio=test_ratio,
                                                                      seed=seed,
                                                                      batch_size=1,
                                                                      mode=mode)

    std_data_test = ConvertGeometricDataSet(loader_g_test)
    std_data_train = ConvertGeometricDataSet(loader_g_train)
    loader_test = TorchDataLoader(std_data_test, batch_size=batch_size, shuffle=False)
    loader_train = TorchDataLoader(std_data_train, batch_size=batch_size, shuffle=True)
    return loader_test, loader_train, num_feat
