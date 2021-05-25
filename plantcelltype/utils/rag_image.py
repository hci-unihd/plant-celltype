from numba import njit, prange
import numpy as np
from numba.core import types
from numba.typed import Dict
from celltype.utils.utils import cantor_sym_pair


def get_neighborhood_structure(x=(1, 1, 1)):
    """
    Build the neighborhood structure array used in the rag generation
    """
    structure = []
    for i in range(-x[0], x[0] + 1):
        for j in range(-x[0], x[0] + 1):
            for k in range(-x[0], x[0] + 1):
                structure.append([i, j, k])
    structure.remove([0, 0, 0])
    return np.array(structure)


@njit(parallel=True)
def compute_edges_image(segmentation, structure):
    
    shape = segmentation.shape
    out_edges = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint)
    label_dict = Dict.empty(key_type=types.int64, value_type=types.int64,)

    for i in prange(1, shape[0] - 1):
        for j in prange(1, shape[1] - 1):
            for k in prange(1, shape[2] - 1):
                _s, is_full = 0, False
                for s in structure:
                    si = s[0] + i
                    sj = s[1] + j
                    sk = s[2] + k
                    
                    s_ijk = segmentation[i, j, k]
                    ss_ijk = segmentation[si, sj, sk] 
                    
                    if s_ijk != ss_ijk:
                        if is_full == False:
                            label_dict = Dict.empty(key_type=types.int64, value_type=types.int64,)
                            
                        label = cantor_sym_pair(s_ijk, ss_ijk)
                        for _key, _value in label_dict.items():
                            if label > 0 and _key == label:
                                label_dict[label] += 1
                                break
                        else:
                            label_dict[label] = 0
                        
                        is_full = True
                    _s += 1
                    
                if is_full:
                    max_count = 0
                    edge_label = 0
                    for _key, _value in label_dict.items():
                        if _value > max_count:
                            edge_label = _key
                            max_count = _value
                            
                    out_edges[i, j, k] = edge_label
                
    return out_edges


@njit(parallel=True)
def _compute_boundaries_image(segmentation, structure):
    shape = segmentation.shape
    out_edges = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint)

    for i in prange(1, shape[0] - 1):
        for j in prange(1, shape[1] - 1):
            for k in prange(1, shape[2] - 1):
                for s in structure:
                    si = s[0] + i
                    sj = s[1] + j
                    sk = s[2] + k

                    s_ijk = segmentation[i, j, k]
                    ss_ijk = segmentation[si, sj, sk]

                    if s_ijk != ss_ijk:
                        out_edges[i, j, k] = 1
                        break
    return out_edges


def compute_boundaries_image(segmentation):
    return _compute_boundaries_image(segmentation, get_neighborhood_structure())


def rectify_edge_image(edge_image, edges):
    """
    This function make sure tha "rag_boundaries" stays consistent with the edges calculated using the elf..compute_rag
    """
    edges_from_image = set(np.unique(edge_image)[1:])
    
    edges_from_rag = []
    for e1, e2 in edges:
        edges_from_rag.append(cantor_sym_pair(e1, e2))
    edges_from_rag = set(edges_from_rag)
    
    if not edges_from_image.issubset(edges_from_rag):
        for value in edges_from_image.difference(edges_from_rag):
            edge_image[edge_image == value] = 0
            
    return edge_image


def boundary_rag_from_seg(segmentation, edges_ids=None):

    rag_boundaries = compute_edges_image(segmentation, get_neighborhood_structure())
    if edges_ids is not None:
        rag_boundaries = rectify_edge_image(rag_boundaries, edges_ids)

    return rag_boundaries
