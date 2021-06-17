import numpy as np
from plantcelltype.utils.io import import_labels_csv
from plantcelltype.utils import cantor_sym_pair
from plantcelltype.utils import create_cell_mapping, map_cell_features2segmentation, create_rag_boundary_from_seg
from plantcelltype.utils.rag_image import rectify_edge_image
from plantcelltype.features.rag import rag_from_seg, get_edges_com_voxels
from plantcelltype.features.edges_features import compute_edges_labels
from plantcelltype.features.cell_features import seg2com, shortest_distance_to_label
from plantcelltype.features.cell_features import compute_cell_volume, compute_cell_surface
from plantcelltype.features.cell_features import rw_betweenness_centrality
from plantcelltype.features.edges_features import compute_edges_length
from plantcelltype.features.sampling import random_points_samples
from plantcelltype.features.utils import make_seg_hollow
from plantcelltype.utils.axis_transforms import find_axis_funiculum, find_label_com


def validate_dict(stack, mandatory_keys, feature_name='feature name'):
    for key in stack.keys():
        assert key in mandatory_keys, f'{key} missing, can not create {feature_name}'


# Base features
def _update_labels_from_csv(stack, csv_path, create_labels_image=True):
    cell_ids, cell_labels = stack['cell_ids'], stack['cell_labels']

    csv_cell_ids, csv_cell_labels = import_labels_csv(csv_path)
    csv_labels_mapping = create_cell_mapping(csv_cell_ids, csv_cell_labels)

    for i, c_ids in enumerate(cell_ids):
        if c_ids in csv_labels_mapping.keys():
            label = csv_labels_mapping[c_ids]
        else:
            print(f"{c_ids} not found in csv file")
            label = 0

        cell_labels[i] = label

    stack['cell_labels'] = cell_labels
    if create_labels_image:
        labels = map_cell_features2segmentation(stack['segmentation'], cell_ids, cell_labels)
        labels = labels.astype('int32')
        stack['labels'] = labels
    return stack


def build_cell_ids(stack, csv_path=None, create_labels_image=True):
    stack['cell_ids'] = np.unique(stack['segmentation'])[1:]
    stack['cell_labels'] = np.zeros_like(stack['cell_ids'])

    if csv_path is not None:
        stack = _update_labels_from_csv(stack, csv_path, create_labels_image=create_labels_image)

    return stack


def build_edges_ids(stack, create_rag_image=True):
    _, edges_ids = rag_from_seg(stack['segmentation'])
    edges_ids = edges_ids.astype('int32')
    stack['edges_ids'] = edges_ids
    edges_labels = compute_edges_labels(stack['cell_ids'], edges_ids, stack['cell_labels'])
    edges_labels = edges_labels.astype('int32')
    stack['edges_labels'] = edges_labels

    if create_rag_image:
        rag_boundaries = create_rag_boundary_from_seg(stack['segmentation'])
        rag_boundaries = rectify_edge_image(rag_boundaries, edges_ids)
        stack['rag_boundaries'] = rag_boundaries

    return stack


def build_basic(stack, csv_path=None):
    stack = build_cell_ids(stack, csv_path=csv_path)
    stack = build_edges_ids(stack)
    return stack


# cell features
def build_cell_com(stack, feat_name='com_voxels', group='cell_features'):
    cell_com = seg2com(stack['segmentation'], stack['cell_ids'])
    cell_com = cell_com.astype('float32')
    stack[group][feat_name] = cell_com
    return stack


def build_hops_to_bg(stack, feat_name='hops_to_bg', group='cell_features'):
    hops_to_bg = shortest_distance_to_label(stack['cell_ids'], stack['edges_ids'])
    hops_to_bg = hops_to_bg.astype('int64')
    stack[group][feat_name] = hops_to_bg
    return stack


def build_volume(stack, feat_name='volume_voxels', group='cell_features'):
    volumes = compute_cell_volume(stack['segmentation'], stack['cell_ids'])
    volumes = volumes.astype('float32')
    stack[group][feat_name] = volumes
    return stack


def build_surface(stack, feat_name='surface_voxels', group='cell_features'):
    surface = compute_cell_surface(stack['segmentation'], stack['rag_boundaries'], stack['cell_ids'])
    surface = surface.astype('float32')
    stack[group][feat_name] = surface
    return stack


def build_rw_centrality(stack, feat_name='rw_centrality', group='cell_features'):
    rw_centrality = rw_betweenness_centrality(stack['cell_ids'], stack['edges_ids'])
    rw_centrality = rw_centrality.astype('float32')
    stack[group][feat_name] = rw_centrality
    return stack


def build_basic_cell_features(stack, group='cell_features'):
    stack[group] = {}
    feat_to_compute = [build_cell_com,
                       build_hops_to_bg,
                       build_volume,
                       build_surface,
                       build_rw_centrality]
    for feat in feat_to_compute:
        stack = feat(stack)
    return stack


# edges features
def build_edges_com_surface(stack, feat_name=('com_voxels', 'surface_voxels'), group='edges_features'):
    rag, edges_ids = rag_from_seg(stack['segmentation'])
    edges_com, edges_surface = get_edges_com_voxels(rag)
    stack[group][feat_name[0]] = edges_com
    stack[group][feat_name[1]] = edges_surface
    return stack


def build_com_distance(stack, axis_transformer, feat_name='com_distance_um', group='edges_features'):
    com_um = axis_transformer.transform_coord(stack['cell_features']['com_voxels'])
    com_distance_um = compute_edges_length(stack['cell_ids'], stack['edges_ids'], com_um)
    com_distance_um = com_distance_um.astype('float32')
    stack[group][feat_name] = com_distance_um
    return stack


def build_basic_edges_features(stack, axis_transformer, group='edges_features'):
    stack[group] = {}
    stack = build_edges_com_surface(stack)
    stack = build_com_distance(stack, axis_transformer)
    return stack


# compute global axis
def build_grs(stack, axis_transformer):
    cell_com_um = axis_transformer.transform_coord(stack['cell_features']['com_voxels'])
    axis = find_axis_funiculum(stack['cell_labels'], cell_com_um)
    center = find_label_com(stack['cell_labels'], cell_com_um, (7, ))
    stack['attributes']['global_reference_system_origin'] = center
    stack['attributes']['global_reference_system_axis'] = axis
    return stack


# compute samples
def build_cell_points_samples(stack, n_points=500, group='cell_samples'):
    stack[group] = {}
    hollow_seg = make_seg_hollow(stack['segmentation'], stack['rag_boundaries'])
    edge_sampling = random_points_samples(hollow_seg, stack['cell_ids'], n_points=n_points)
    stack[group]['random_samples'] = edge_sampling
    return stack


def build_edges_points_samples(stack, n_points=50, group='edges_samples'):
    stack[group] = {}
    rag_image_ct1 = create_rag_boundary_from_seg(stack['segmentation'], stack['edges_ids'], min_counts=1)
    cantor_ids = np.array([cantor_sym_pair(e1, e2) for e1, e2 in stack['edges_ids']])
    edge_sampling = random_points_samples(rag_image_ct1, cantor_ids, n_points=n_points)
    stack[group]['random_samples'] = edge_sampling
    return stack
