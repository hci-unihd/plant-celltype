import numpy as np
import pandas as pd
import tqdm

from plantcelltype.graphnn.data_loader import collect_cell_features
from plantcelltype.utils import open_full_stack


def _make_data_onedim_dict(dict_data):
    new_dict_data = {}
    for key, value in dict_data.items():

        if value.ndim == 1:
            new_dict_data[key] = value
        elif value.ndim == 2 and value.shape[1] == 3:
            new_keys = [f'{key}_{ax}' for ax in ['x', 'y', 'z']]
            for i, new_key in enumerate(new_keys):
                new_dict_data[new_key] = value[:, i]

    return new_dict_data


def stack_to_df(stack, at,
                global_feat=('cell_ids', 'cell_labels'),
                attributes=('dataset', 'element_size_um', 'stack', 'stage')):

    cell_features = stack['cell_features']
    cell_features['com_grs'] = at.transform_coord(cell_features['com_voxels'])
    cell_features['volume_grs'] = at.scale_volumes(cell_features['volume_voxels'])
    cell_features['surface_grs'] = at.scale_volumes(cell_features['surface_voxels'])

    dict_data = {}
    for g_feat in global_feat:
        dict_data[g_feat] = stack[g_feat]

    cell_feat = _make_data_onedim_dict(cell_features)
    dict_data.update(cell_feat)

    num_feat = stack[global_feat[0]].shape[0]
    for attr in attributes:
        dict_data[attr] = num_feat * [stack['attributes'][attr]]

    df_data = pd.DataFrame(dict_data)
    return df_data


def multi_stack_df(list_files):
    glob_df_stack = pd.DataFrame()
    for file in tqdm.tqdm(list_files):
        stack, at = open_full_stack(file, ['cell_ids', 'cell_labels', 'cell_features'])
        df_stack = stack_to_df(stack, at)
        glob_df_stack = glob_df_stack.append(df_stack)
    return glob_df_stack


def collect_multi_features(list_files):
    list_feat_vector = []
    for file in tqdm.tqdm(list_files):
        stack, at = open_full_stack(file, ['cell_ids', 'cell_labels', 'cell_features'])
        feat_vector = collect_cell_features(stack, at)
        list_feat_vector.append(feat_vector)

    return np.concatenate(list_feat_vector, axis=0)
