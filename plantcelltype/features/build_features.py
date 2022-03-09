import numpy as np
from skspatial.objects import Vector

from plantcelltype.features.cell_features import compute_cell_volume, compute_cell_surface
from plantcelltype.features.cell_features import compute_cell_average_edt, get_es_com, get_proposed_es
from plantcelltype.features.cell_features import compute_rw_betweenness_centrality, compute_degree_centrality
from plantcelltype.features.cell_features import seg2com, shortest_distance_to_label, compute_pca, compute_pca_comp_idx
from plantcelltype.features.cell_vector_features import get_vectors_orientation_mapping
from plantcelltype.features.cell_vector_features import compute_sym_length_along_cell_axis
from plantcelltype.features.cell_vector_features import compute_proj_length_on_sphere
from plantcelltype.features.cell_vector_features import compute_local_reference_axis1
from plantcelltype.features.cell_vector_features import compute_local_reference_axis2
from plantcelltype.features.cell_vector_features import compute_local_reference_axis3
from plantcelltype.features.clean_segmentation import remove_disconnected_components, unique_seg
from plantcelltype.features.clean_segmentation import set_label_to_bg, size_filter_bg_preserving
from plantcelltype.features.edges_features import compute_edges_labels, compute_edges_length
from plantcelltype.features.edges_vector_features import compute_edges_planes
from plantcelltype.features.rag import rag_from_seg, get_edges_com_voxels
from plantcelltype.features.sampling import compute_random_points_samples
from plantcelltype.features.sampling import compute_farthest_points_samples
from plantcelltype.features.utils import make_seg_hollow
from plantcelltype.utils import cantor_sym_pair
from plantcelltype.utils import create_cell_mapping, map_cell_features2segmentation, create_rag_boundary_from_seg
from plantcelltype.utils.axis_transforms import find_axis_funiculus, find_label_com
from plantcelltype.utils.io import import_labels_csv
from plantcelltype.utils.rag_image import rectify_edge_image


# Base features_importance
def build_labels_from_csv(stack, csv_path, create_labels_image=True):
    """update labels from csv and optionally creat a gt image"""
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

    # create label image
    if create_labels_image:
        labels = map_cell_features2segmentation(stack['segmentation'], cell_ids, cell_labels)
        labels = labels.astype('int32')
        stack['labels'] = labels
    return stack


def build_cell_ids(stack, csv_path=None, create_labels_image=True):
    """create ids and labels (if available)"""
    stack['cell_ids'], _ = unique_seg(stack['segmentation'])
    stack['cell_labels'] = np.zeros_like(stack['cell_ids'])

    if csv_path is not None:
        stack = build_labels_from_csv(stack, csv_path, create_labels_image=create_labels_image)

    return stack


def build_edges_ids(stack, create_rag_image=True):
    """create graph_features and edges labels (if available)"""
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


def build_preprocessing(stack, size_filter=50, label=None):
    segmentation = stack['segmentation']
    if label is None:
        # here numpy unique is required to ensure the bg is correctly set
        label = np.unique(segmentation)[0]

    segmentation = set_label_to_bg(segmentation, label)
    segmentation = size_filter_bg_preserving(segmentation, size_filter)
    segmentation = remove_disconnected_components(segmentation)

    stack['segmentation'] = segmentation
    return stack


def build_basic(stack, csv_path=None):
    stack = build_cell_ids(stack, csv_path=csv_path)
    stack = build_edges_ids(stack)
    return stack


# cell features_importance
def build_cell_com(stack, feat_name='com_voxels', group='cell_features'):
    cell_com = seg2com(stack['segmentation'], stack['cell_ids'])
    cell_com = cell_com.astype('float32')
    stack[group][feat_name] = cell_com
    return stack


def build_hops_to_bg(stack, feat_name='hops_to_bg', group='cell_features'):
    hops_to_bg = shortest_distance_to_label(stack['cell_ids'], stack['edges_ids'])
    hops_to_bg = hops_to_bg.astype('int32')
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
    rw_centrality = compute_rw_betweenness_centrality(stack['cell_ids'], stack['edges_ids'])
    rw_centrality = rw_centrality.astype('float32')
    stack[group][feat_name] = rw_centrality
    return stack


def build_degree_centrality(stack, feat_name='degree_centrality', group='cell_features'):
    degree_centrality = compute_degree_centrality(stack['cell_ids'], stack['edges_ids'])
    degree_centrality = degree_centrality.astype('float32')
    stack[group][feat_name] = degree_centrality
    return stack


def build_average_cell_edt(stack, label=0, feat_name='bg_edt_um', group='cell_features'):
    edt_um = compute_cell_average_edt(stack['cell_ids'],
                                      stack['segmentation'],
                                      voxel_size=stack['attributes']['element_size_um'],
                                      label=label)
    stack[group][feat_name] = edt_um.astype('float32')
    return stack


def build_basic_cell_features(stack, group='cell_features'):
    stack[group] = {}
    feat_to_compute = [build_cell_com,
                       build_hops_to_bg,
                       build_volume,
                       build_surface,
                       build_rw_centrality,
                       build_degree_centrality,
                       build_average_cell_edt]

    for feat in feat_to_compute:
        stack = feat(stack)
    return stack


# propose es
def build_es_proposal(stack, es_label=8):
    es_index, es_ids = get_proposed_es(stack, es_label)

    com_voxels = stack['cell_features']['com_voxels']
    stack['attributes']['es_index'] = es_index
    stack['attributes']['es_label'] = es_ids
    stack['attributes']['es_com_voxels'] = [com_voxels[_es_index].tolist() for _es_index in es_index]
    return stack


def build_es_features(stack, compute_etd=False, feat_name=('hops_to_es', 'edt_es_um'), group='cell_features'):
    es_label = stack['attributes']['es_label']
    sd = shortest_distance_to_label(stack['cell_ids'],
                                    stack['edges_ids'],
                                    label=es_label,
                                    not_bg=True)
    stack[group][feat_name[0]] = sd.astype('int32')

    if compute_etd:
        edt = compute_cell_average_edt(stack['cell_ids'],
                                       stack['segmentation'],
                                       voxel_size=stack['attributes']['element_size_um'],
                                       label=es_label)
        stack[group][feat_name[1]] = edt.astype('float32')
    return stack


# edges features_importance
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
def build_grs_from_labels(stack,
                          axis_transformer,
                          label=7,
                          group='attributes',
                          feat_name=('global_reference_system_origin', 'global_reference_system_axis')):
    cell_com_um = axis_transformer.transform_coord(stack['cell_features']['com_voxels'])
    axis = find_axis_funiculus(stack['cell_labels'], cell_com_um)
    center = find_label_com(stack['cell_labels'], cell_com_um, (label,))
    stack[group][feat_name[0]] = center
    stack[group][feat_name[1]] = axis
    return stack


def build_grs_from_labels_funiculus(stack, axis_transformer, **kwargs):
    return build_grs_from_labels(stack=stack,
                                 axis_transformer=axis_transformer,
                                 label=7,
                                 **kwargs)


def build_grs_from_labels_surface(stack, axis_transformer, **kwargs):
    return build_grs_from_labels(stack=stack,
                                 axis_transformer=axis_transformer,
                                 label=1,
                                 **kwargs)


def build_trivial_grs(stack,
                      axis_transformer,
                      group='attributes',
                      feat_name=('global_reference_system_origin', 'global_reference_system_axis')):
    stack['attributes']['global_reference_system_origin'] = (0, 0, 0)
    stack['attributes']['global_reference_system_axis'] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    return stack


def build_es_trivial_grs(stack,
                         axis_transformer,
                         es_label=8,
                         group='attributes',
                         feat_name=('global_reference_system_origin', 'global_reference_system_axis')):
    stack[group][feat_name[0]] = get_es_com(stack, axis_transformer, es_label)
    stack[group][feat_name[1]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    return stack


def build_es_pca_grs(stack,
                     axis_transformer,
                     es_label=8,
                     group='attributes',
                     feat_name=('global_reference_system_origin', 'global_reference_system_axis')):
    es_index, es_ids = get_proposed_es(stack, es_label)
    masks = []
    for _es_ids in es_ids:
        masks.append(stack['segmentation'] == _es_ids)

    masks = np.logical_or.reduce(masks)
    samples_voxels = np.stack(np.nonzero(masks)).T
    samples_grs = axis_transformer.transform_coord(samples_voxels)
    components, _ = compute_pca_comp_idx(samples_grs)

    stack[group][feat_name[0]] = get_es_com(stack, axis_transformer, es_label)
    stack[group][feat_name[1]] = components
    return stack


# compute local axis
def build_edges_planes(stack, axis_transform):
    edge_sampling_grs = stack['edges_samples']['fps_samples_grs']
    cell_com_grs = axis_transform.transform_coord(stack['cell_features']['com_voxels'])
    edges_com_grs = axis_transform.transform_coord(stack['edges_features']['com_voxels'])
    origin = axis_transform.transform_coord([0, 0, 0])

    edges_planes = compute_edges_planes(stack['cell_ids'],
                                        stack['edges_ids'],
                                        cell_com_grs,
                                        edge_sampling_grs,
                                        edges_com_grs,
                                        origin)

    stack["edges_features"]["plane_vectors_grs"] = edges_planes.astype('float32')
    return stack


def build_lrs(stack, axis_transform, global_axis=0, group='cell_features'):
    cell_com_grs = axis_transform.transform_coord(stack['cell_features']['com_voxels'])
    lrs_axis_1 = compute_local_reference_axis1(stack['cell_ids'],
                                               stack['edges_ids'],
                                               cell_hops_to_bg=stack['cell_features']['hops_to_bg'],
                                               cell_com=cell_com_grs,
                                               edges_plane_vectors=stack['edges_features']['plane_vectors_grs'])

    cell_com_grs = axis_transform.transform_coord(stack['cell_features']['com_voxels'])

    lrs_axis_2, lrs_axis_2_angle = compute_local_reference_axis2(stack['cell_ids'],
                                                                 stack['edges_ids'],
                                                                 cell_com_grs,
                                                                 stack['cell_features']['hops_to_bg'],
                                                                 global_axis=axis_transform.axis[global_axis])

    lrs_axis_3 = compute_local_reference_axis3(lrs_axis_1, lrs_axis_2)

    stack[group]['lrs_axis1_grs'] = lrs_axis_1.astype('float32')
    stack[group]['lrs_axis2_grs'] = lrs_axis_2.astype('float32')
    stack[group]['lrs_axis3_grs'] = lrs_axis_3.astype('float32')

    stack[group]['lrs_axis12_dot_grs'] = np.sum(lrs_axis_1 * lrs_axis_2, axis=1).astype('float32')
    stack[group]['lrs_axis2_angle_grs'] = lrs_axis_2_angle.astype('float32')
    return stack


# compute samples
def build_abstract_random_samples(stack, segmentation, cell_ids,
                                  n_points=500, seed=0,
                                  feat_name='random_samples', group='cell_samples'):
    if group not in stack:
        stack[group] = {}

    samples = compute_random_points_samples(segmentation,
                                            cell_ids,
                                            n_points=n_points,
                                            seed=seed)
    stack[group][feat_name] = samples
    return stack


def build_hollow_cell_points_random_samples(stack, n_points=500, seed=0, group='cell_samples'):
    hollow_seg = make_seg_hollow(stack['segmentation'], stack['rag_boundaries'])
    return build_abstract_random_samples(stack, hollow_seg, stack['cell_ids'],
                                         n_points=n_points, seed=seed,
                                         feat_name='hollow_random_samples_voxels', group=group)


def build_cell_points_random_samples(stack, n_points=500, seed=0, group='cell_samples'):
    return build_abstract_random_samples(stack, stack['segmentation'], stack['cell_ids'],
                                         n_points=n_points, seed=seed,
                                         feat_name='random_samples_voxels', group=group)


def build_edges_points_random_samples(stack, n_points=500, seed=0, group='edges_samples'):
    cantor_ids = np.array([cantor_sym_pair(e1, e2) for e1, e2 in stack['edges_ids']])
    return build_abstract_random_samples(stack, stack['rag_boundaries'], cantor_ids,
                                         n_points=n_points, seed=seed,
                                         feat_name='random_samples_voxels', group=group)


def build_abstract_fps_samples(stack, segmentation, cell_ids,
                               axis_transformer=None, n_points=500, seed=0,
                               feat_name='random_samples', group='cell_samples'):
    if group not in stack:
        stack[group] = {}

    if axis_transformer is None:
        grs_key, pos_transform = 'voxels', None
    else:
        grs_key, pos_transform = 'grs', axis_transformer.transform_coord

    samples = compute_farthest_points_samples(segmentation,
                                              cell_ids,
                                              n_points=n_points,
                                              seed=seed,
                                              pos_transform=pos_transform)

    stack[group][f'{feat_name}_{grs_key}'] = samples
    return stack


def build_hollow_cell_points_fps_samples(stack, axis_transformer=None, n_points=500, seed=0, group='cell_samples'):
    hollow_seg = make_seg_hollow(stack['segmentation'], stack['rag_boundaries'])
    return build_abstract_fps_samples(stack, hollow_seg, stack['cell_ids'],
                                      axis_transformer=axis_transformer, n_points=n_points, seed=seed,
                                      feat_name='hollow_fps_samples', group=group)


def build_cell_points_fps_samples(stack, axis_transformer=None, n_points=500, seed=0, group='cell_samples'):
    return build_abstract_fps_samples(stack, stack['segmentation'], stack['cell_ids'],
                                      axis_transformer=axis_transformer, n_points=n_points, seed=seed,
                                      feat_name='fps_samples', group=group)


def build_edges_points_fps_samples(stack, axis_transformer=None, n_points=500, seed=0, group='edges_samples'):
    cantor_ids = np.array([cantor_sym_pair(e1, e2) for e1, e2 in stack['edges_ids']])
    return build_abstract_fps_samples(stack, stack['rag_boundaries'], cantor_ids,
                                      axis_transformer=axis_transformer, n_points=n_points, seed=seed,
                                      feat_name='fps_samples', group=group)


# compute PCA along axis
def build_pca_features(stack, axis_transformer, group='cell_features'):
    origin_grs = axis_transformer.transform_coord((0, 0, 0))
    samples_grs = stack['cell_samples']['hollow_fps_samples_grs']
    pca1, pca2, pca3, pca_v = compute_pca(samples_grs, origin_grs)

    stack[group]['pca_axis1_grs'] = pca1.astype('float32')
    stack[group]['pca_axis2_grs'] = pca2.astype('float32')
    stack[group]['pca_axis3_grs'] = pca3.astype('float32')

    stack[group]['pca_explained_variance_grs'] = pca_v.astype('float32')
    return stack


# compute length along axis
def build_length_along_axis(stack,
                            axis_transformer,
                            group='cell_features',
                            axis_name=('lrs_axis1_grs', 'lrs_axis2_grs', 'lrs_axis3_grs'),
                            feat_name=('length_axis1_grs', 'length_axis2_grs', 'length_axis3_grs')):
    origin_grs = axis_transformer.transform_coord((0, 0, 0)).astype('float32')
    com_grs = axis_transformer.transform_coord(stack[group]['com_voxels']).astype('float32')
    samples_grs = stack['cell_samples']['hollow_fps_samples_grs'].astype('float32')

    for _axis, _feat in zip(axis_name, feat_name):
        in_feat = stack[group][_axis]
        len_axis = compute_sym_length_along_cell_axis(in_feat,
                                                      com_grs,
                                                      samples_grs,
                                                      origin=origin_grs)
        stack[group][_feat] = len_axis.astype('float32')

    return stack


def build_length_along_local_axis(stack, axis_transformer, group='cell_features'):
    return build_length_along_axis(stack, axis_transformer, group)


def build_length_along_pca_axis(stack, axis_transformer, group='cell_features'):
    stack = build_length_along_axis(stack,
                                    axis_transformer,
                                    group,
                                    axis_name=('pca_axis1_grs', 'pca_axis2_grs', 'pca_axis3_grs'),
                                    feat_name=('pca_length_axis1_grs', 'pca_length_axis2_grs', 'pca_length_axis3_grs'))
    return stack


def build_cell_dot_features(stack, axis_transformer, group='cell_features'):
    cell_features, global_axis = stack['cell_features'], axis_transformer.axis

    # proj cell com proj on the global axis
    cell_com_grs = axis_transformer.transform_coord(cell_features['com_voxels'])
    cell_com_grs = cell_com_grs
    stack[group]['com_proj_grs'] = cell_com_grs.dot(global_axis.T).astype('float32')

    # proj cell lrs axis proj on the global axis
    for axis_mode in ['lrs', 'pca']:
        for axis in [1, 2, 3]:
            feat_name = f'{axis_mode}_proj_axis{axis}_grs'
            proj_axis = cell_features[f'{axis_mode}_axis1_grs'].dot(global_axis.T)
            stack[group][feat_name] = proj_axis.astype('float32')
    return stack


def build_cell_orientation_features(stack, group='cell_features'):
    cell_features = stack['cell_features']
    # proj cell lrs axis proj on the global axis
    for axis_mode in ['lrs', 'pca']:
        for axis in [1, 2, 3]:
            feat_name = f'{axis_mode}_orientation_axis{axis}_grs'
            orientation_vectors = get_vectors_orientation_mapping(cell_features[f'{axis_mode}_axis{axis}_grs'])
            stack[group][feat_name] = orientation_vectors.astype('float32')
    return stack


def get_edges_dot(e1, e2, axis_mapping):
    e1_axis = axis_mapping[e1]
    e2_axis = axis_mapping[e2]
    return np.dot(e1_axis, e2_axis)


def build_edges_dot_features(stack, axis_transformer, group='edges_features'):
    cell_features, global_axis = stack['cell_features'], axis_transformer.axis
    cell_com_grs = axis_transformer.transform_coord(cell_features['com_voxels'])

    # create mapping
    cell_com_grs = create_cell_mapping(stack['cell_ids'], cell_com_grs)
    cell_axis1_grs = create_cell_mapping(stack['cell_ids'], cell_features['lrs_axis1_grs'])
    cell_axis2_grs = create_cell_mapping(stack['cell_ids'], cell_features['lrs_axis2_grs'])
    cell_axis3_grs = create_cell_mapping(stack['cell_ids'], cell_features['lrs_axis3_grs'])

    # init feat
    (lrs_dot_axis1_grs,
     lrs_dot_axis2_grs,
     lrs_dot_axis3_grs,
     lrs1e1_dot_ev_grs,
     lrs2e1_dot_ev_grs,
     lrs3e1_dot_ev_grs,
     lrs1e2_dot_ev_grs,
     lrs2e2_dot_ev_grs,
     lrs3e2_dot_ev_grs) = [np.zeros(stack['edges_ids'].shape[0]) for _ in range(9)]

    lrs_proj_grs = np.zeros((stack['edges_ids'].shape[0], 3))

    for i, (e1, e2) in enumerate(stack['edges_ids']):
        if e1 > 0 and e2 > 0:
            # lrs*_e1 dot lrs*_e2
            lrs_dot_axis1_grs[i] = get_edges_dot(e1, e2, cell_axis1_grs)
            lrs_dot_axis2_grs[i] = get_edges_dot(e1, e2, cell_axis2_grs)
            lrs_dot_axis3_grs[i] = get_edges_dot(e1, e2, cell_axis3_grs)

            e_v = Vector.from_points(cell_com_grs[e1], cell_com_grs[e2]).unit()

            # lrs* dot grs*
            lrs_proj_grs[i] = np.dot(e_v, global_axis.T)

            # lrs*e1 dot ev
            lrs1e1_dot_ev_grs[i] = np.dot(e_v, cell_axis1_grs[e1])
            lrs2e1_dot_ev_grs[i] = np.dot(e_v, cell_axis2_grs[e1])
            lrs3e1_dot_ev_grs[i] = np.dot(e_v, cell_axis3_grs[e1])

            # lrs*e2 dot ev
            lrs1e2_dot_ev_grs[i] = np.dot(e_v, cell_axis1_grs[e2])
            lrs2e2_dot_ev_grs[i] = np.dot(e_v, cell_axis2_grs[e2])
            lrs3e2_dot_ev_grs[i] = np.dot(e_v, cell_axis3_grs[e2])

    # create attr
    # lrs*_e1 dot lrs*_e2
    stack[group]['lrs_dot_axis1_grs'] = lrs_dot_axis1_grs.astype('float32')
    stack[group]['lrs_dot_axis2_grs'] = lrs_dot_axis2_grs.astype('float32')
    stack[group]['lrs_dot_axis3_grs'] = lrs_dot_axis3_grs.astype('float32')

    # lrs* dot grs*
    stack[group]['lrs_proj_grs'] = lrs_proj_grs.astype('float32')

    # lrs*e1 dot ev
    stack[group]['lrs1e1_dot_ev_grs'] = lrs1e1_dot_ev_grs.astype('float32')
    stack[group]['lrs2e1_dot_ev_grs'] = lrs2e1_dot_ev_grs.astype('float32')
    stack[group]['lrs3e1_dot_ev_grs'] = lrs3e1_dot_ev_grs.astype('float32')

    # lrs*e2 dot ev
    stack[group]['lrs1e2_dot_ev_grs'] = lrs1e2_dot_ev_grs.astype('float32')
    stack[group]['lrs2e2_dot_ev_grs'] = lrs2e2_dot_ev_grs.astype('float32')
    stack[group]['lrs3e2_dot_ev_grs'] = lrs3e2_dot_ev_grs.astype('float32')

    return stack


# precompute grs features_importance
def build_cell_transformed_voxels_features(stack, axis_transform, group='cell_features'):
    stack[group]['com_grs'] = axis_transform.transform_coord(stack[group]['com_voxels'])
    stack[group]['volume_um'] = axis_transform.scale_volumes(stack[group]['volume_voxels'])
    stack[group]['surface_um'] = axis_transform.scale_volumes(stack[group]['surface_voxels'])
    return stack


def build_edges_transformed_voxels_features(stack, axis_transform, group='edges_features'):
    stack[group]['com_grs'] = axis_transform.transform_coord(stack[group]['com_voxels'])
    stack[group]['surface_um'] = axis_transform.scale_volumes(stack[group]['surface_voxels'])
    return stack


def build_proj_length_on_sphere(stack,
                                axis_transformer,
                                n_samples=64,
                                group='cell_features'):
    origin_grs = axis_transformer.transform_coord((0, 0, 0)).astype('float32')
    com_grs = axis_transformer.transform_coord(stack[group]['com_voxels']).astype('float32')
    samples_grs = stack['cell_samples']['hollow_fps_samples_grs'].astype('float32')
    stack[group]['proj_length_unit_sphere'] = compute_proj_length_on_sphere(com_grs,
                                                                            samples_grs,
                                                                            n_samples=n_samples,
                                                                            origin=origin_grs)
    return stack
