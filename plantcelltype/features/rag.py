import networkx as nx
import numpy as np
from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length

from plantcelltype.utils.utils import create_cell_mapping


def rag_from_seg(segmentation):
    rag = compute_rag(segmentation)
    edges = rag.uvIds()
    return rag, edges


def remove_bg_from_edges_ids(edges_ids, bg=0):
    mask = np.where(np.min(edges_ids, axis=1) != bg)[0]
    return edges_ids[mask]


def fill_bg_from_edges_ids(edges_ids, edges_features, value=1, bg=0):
    mask = np.where(np.min(edges_ids, axis=1) != bg)[0]
    new_features = np.zeros(edges_ids.shape[0])
    new_features[~mask] = value
    new_features[mask] = edges_features
    return new_features


def build_nx_graph(cell_ids, edges_ids, cell_com=None, remove_bg=True):
    ovule_graph = nx.Graph()

    if cell_com is not None:
        cell_com_mapping = create_cell_mapping(cell_ids, cell_com)
    else:
        cell_com_mapping = create_cell_mapping(cell_ids, np.ones_like(cell_ids))

    edges_ids = remove_bg_from_edges_ids(edges_ids) if remove_bg else edges_ids

    for (e1, e2) in edges_ids:
        d = np.sqrt(np.sum((cell_com_mapping[e1] - cell_com_mapping[e2]) ** 2)) if remove_bg else 1
        ovule_graph.add_edge(e1, e2, weight=d)

    return ovule_graph


def get_edges_com_voxels(rag):
    coord = [np.arange(_s) for _s in rag.shape]
    coord_mg = np.meshgrid(*coord, indexing='ij')
    edges_com_voxels = np.stack([compute_boundary_mean_and_length(rag, _coord_mg.astype(np.float32))
                                 for _coord_mg in coord_mg])
                            
    return edges_com_voxels[:, :, 0].T, edges_com_voxels[0, :, 1]  # return (edges com, edges length)


def rectify_rag_names(cell_ids, edges_ids):
    cell_ids_mapping = create_cell_mapping(cell_ids, np.arange(cell_ids.shape[0]))

    edges_ids = remove_bg_from_edges_ids(edges_ids)
    new_edges_ids = [[cell_ids_mapping[e1], cell_ids_mapping[e2]] for e1, e2 in edges_ids]
    return np.array(new_edges_ids)

