import numpy as np
from networkx.algorithms.centrality import current_flow_betweenness_centrality
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, eigenvector_centrality
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt
from numba import njit
from numba import types
from numba.typed import Dict

from plantcelltype.features.rag import build_nx_graph, remove_bg_from_edges_ids
from plantcelltype.features.sampling import farthest_points_sampling
from plantcelltype.features.utils import label2com, make_seg_hollow, check_valid_idx

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
    return distance[cell_ids]


def compute_max_diameter(segmentation, cell_ids, cell_com, rag_boundaries=None):
    if rag_boundaries is not None:
        segmentation = make_seg_hollow(segmentation, rag_boundaries)
    edges_samples = farthest_points_sampling(segmentation, cell_ids, cell_com, n_points=2)
    distance = np.sqrt(np.sum((edges_samples[:, 1] - edges_samples[:, 2])**2, axis=1))
    return distance


def compute_cell_volume(segmentation, cell_ids):
    _cell_id, volumes = np.unique(segmentation, return_counts=True)
    assert np.allclose(_cell_id[1:], cell_ids)
    return volumes[1:]


def compute_cell_surface(segmentation, rag_boundaries, cell_ids):
    h_seg = make_seg_hollow(segmentation, rag_boundaries)
    h_idx, h_counts = np.unique(h_seg, return_counts=True)
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


def compute_generic_centrality(cell_ids, edges_ids, centrality='degree', cell_com=None, args=()):
    nx_graph = build_nx_graph(cell_ids, edges_ids, cell_com)
    centrality_mapping = _supported_centrality[centrality](nx_graph, *args)
    centrality = np.zeros(cell_ids.shape[0])
    for i, _idx in enumerate(cell_ids):
        centrality[i] = centrality_mapping[_idx]

    return centrality


def compute_degree_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='degree', cell_com=None)


def compute_rw_betweenness_centrality(cell_ids, edges_ids, cell_com=None):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='rw_betweenness', cell_com=cell_com)


def compute_betweenness_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='betweenness', cell_com=None)


def compute_eigen_vector_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='eigen_vector', args=None, cell_com=None)


def compute_pca_comp_idx(cell_samples):
    pca = PCA()
    pca.fit(cell_samples)
    return pca.components_, pca.explained_variance_


def compute_pca(cell_samples, origin=(0, 0, 0)):
    pca_axis_1, pca_axis_2, pca_axis_3 = [], [], []
    pca_explained_variance = []

    for samples in cell_samples:
        valid_idx, _ = check_valid_idx(samples, origin)
        valid_samples = samples[valid_idx]
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
