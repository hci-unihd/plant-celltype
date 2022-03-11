import numpy as np
from networkx.algorithms.centrality import current_flow_betweenness_centrality
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, eigenvector_centrality
from numba import njit
from numba import types
from numba.typed import Dict
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA

from plantcelltype.features.rag import build_nx_graph, remove_bg_from_edges_ids
from plantcelltype.features.clean_segmentation import unique_seg
from plantcelltype.features.utils import label2com, make_seg_hollow, check_valid_idx
from plantcelltype.utils.axis_transforms import find_label_com

_supported_centrality = {'degree': degree_centrality,
                         'betweenness': betweenness_centrality,
                         'rw_betweenness': current_flow_betweenness_centrality,
                         'eigenvector': eigenvector_centrality}


def seg2com(segmentation, cell_ids):
    return label2com(segmentation, cell_ids)


def shortest_distance_to_label(cell_ids, edges_ids, label=0, not_bg=False):
    if not_bg:
        edges_ids = remove_bg_from_edges_ids(edges_ids)

    adj = csr_matrix((np.ones(edges_ids.shape[0]),
                      (edges_ids[:, 0], edges_ids[:, 1])),
                     shape=(cell_ids.max() + 1, cell_ids.max() + 1))
    adj = adj + adj.T
    distance = shortest_path(adj, indices=label)
    if distance.ndim == 2:
        distance = np.min(distance, axis=0)

    return distance[cell_ids]


def compute_cell_volume(segmentation, cell_ids):
    _cell_id, volumes = unique_seg(segmentation)
    assert np.allclose(_cell_id, cell_ids)
    return volumes


def compute_cell_surface(segmentation, rag_boundaries, cell_ids):
    h_seg = make_seg_hollow(segmentation, rag_boundaries)
    h_idx, h_counts = unique_seg(h_seg)
    surface_mapping = {}
    for _key, _value in zip(h_idx, h_counts):
        surface_mapping[_key] = _value

    surface_array = np.zeros(cell_ids.shape[0])
    for i, idx in enumerate(cell_ids):
        _value = surface_mapping.get(idx, 1)
        surface_array[i] = _value
    return surface_array


def compute_sphericity(cell_volume, cell_surface):
    sphericity = np.pi ** (1/3) * (6 * cell_volume) ** (2 / 3)
    sphericity /= cell_surface
    return sphericity


@njit()
def _cell_average_edt(label_image, dt_image, label_ids):
    shape = label_image.shape
    cell_mapping = Dict.empty(key_type=types.int64, value_type=types.int64)

    for i, _ids in enumerate(label_ids):
        cell_mapping[_ids] = i

    edt_vector = np.zeros((label_ids.shape[0]))
    counts_vector = np.zeros((label_ids.shape[0]))

    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                _label = label_image[i, j, k]
                if _label != 0:
                    _ids = cell_mapping[_label]
                    edt_vector[_ids] += dt_image[i, j, k]
                    counts_vector[_ids] += 1

    for i, _ids in enumerate(label_ids):
        edt_vector[i] /= counts_vector[i]

    return edt_vector


def compute_cell_average_edt(cell_ids, segmentation, voxel_size=(1, 1, 1), label=0):
    mask = segmentation != label
    dt_mask = distance_transform_edt(mask, sampling=voxel_size)
    return _cell_average_edt(segmentation, dt_mask, cell_ids)


def compute_generic_centrality(cell_ids, edges_ids, centrality='degree', cell_com=None, kwargs=None):
    nx_graph = build_nx_graph(cell_ids, edges_ids, cell_com)
    kwargs = {} if kwargs is None else kwargs
    centrality_mapping = _supported_centrality[centrality](nx_graph, **kwargs)
    centrality = np.zeros(cell_ids.shape[0])
    for i, _idx in enumerate(cell_ids):
        centrality[i] = centrality_mapping[_idx]
    return centrality


def compute_degree_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='degree')


def compute_rw_betweenness_centrality(cell_ids, edges_ids, cell_com=None):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='rw_betweenness', cell_com=cell_com)


def compute_betweenness_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='betweenness')


def compute_eigen_vector_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='eigen_vector')


def compute_pca_comp_idx(cell_samples):
    pca = PCA()
    pca.fit(cell_samples)
    return pca.components_, pca.explained_variance_ratio_


def compute_pca(cell_samples, origin=(0, 0, 0)):
    pca_axis_1, pca_axis_2, pca_axis_3 = [], [], []
    pca_explained_variance = []

    for samples in cell_samples:
        valid_samples, _ = check_valid_idx(samples, origin)
        components, explained_variance = compute_pca_comp_idx(valid_samples)
        pca_axis_1.append(components[0])
        pca_axis_2.append(components[1])
        pca_axis_3.append(components[2])

        pca_explained_variance.append(explained_variance)

    pca_axis_1 = np.array(pca_axis_1)
    pca_axis_2 = np.array(pca_axis_2)
    pca_axis_3 = np.array(pca_axis_3)

    pca_explained_variance = np.array(pca_explained_variance)
    return pca_axis_1, pca_axis_2, pca_axis_3, pca_explained_variance


def get_es_com(stack, axis_transformer, es_label=8):
    cell_com_um = axis_transformer.transform_coord(stack['cell_features']['com_voxels'])
    es_com_voxels = find_label_com(stack['cell_labels'], cell_com_um, (es_label,))
    if es_com_voxels[0] is None:
        es_com_voxels = axis_transformer.transform_coord(stack['attributes']['es_com_voxels'][0])
    return es_com_voxels


def get_proposed_es(stack, es_label=8):
    es_index = np.where(stack['cell_labels'] == es_label)[0]
    if len(es_index) < 1:
        es_index = [np.argmax(stack['cell_features']['volume_voxels'])]

    es_ids = stack['cell_ids'][es_index]
    return es_index, es_ids
