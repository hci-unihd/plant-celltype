import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from plantcelltype.features.utils import label2com, make_seg_hollow
from plantcelltype.features.rag import build_nx_graph
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, eigenvector_centrality
from networkx.algorithms.centrality import current_flow_betweenness_centrality
from plantcelltype.features.sampling import farthest_points_sampling

_supported_centrality = {'degree': degree_centrality,
                         'betweenness': betweenness_centrality,
                         'rw_betweenness': current_flow_betweenness_centrality,
                         'eigenvector': eigenvector_centrality}


def seg2com(segmentation, cell_ids):
    return label2com(segmentation, cell_ids)


def shortest_distance_to_label(cell_ids, edges_ids, label=0):
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


def compute_generic_centrality(cell_ids, edges_ids, centrality='degree', cell_com=None, args=()):
    nx_graph = build_nx_graph(cell_ids, edges_ids, cell_com)
    centrality_mapping = _supported_centrality[centrality](nx_graph, *args)
    centrality = np.zeros(cell_ids.shape[0])
    for i, _idx in enumerate(cell_ids):
        centrality[i] = centrality_mapping[_idx]

    return centrality


def compute_degree_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='degree', cell_com=None)


def rw_betweenness_centrality(cell_ids, edges_ids, cell_com=None):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='rw_betweenness', cell_com=cell_com)


def betweenness_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='betweenness', cell_com=None)


def eigen_vector_centrality(cell_ids, edges_ids):
    return compute_generic_centrality(cell_ids, edges_ids, centrality='eigen_vector', args=None, cell_com=None)
