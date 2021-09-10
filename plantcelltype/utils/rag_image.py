import numpy as np
from numba import njit, prange
from numba import types
from numba.typed import Dict

from plantcelltype.utils.utils import cantor_sym_pair


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
def __create_rag_boundary_image(segmentation, structure, min_counts=0):
    shape = segmentation.shape
    out_edges = np.zeros((shape[0], shape[1], shape[2]), dtype='int64')
    struct_label_array = np.zeros(structure.shape[0])

    for i in prange(1, shape[0] - 1):
        for j in prange(1, shape[1] - 1):
            for k in prange(1, shape[2] - 1):
                _s, is_full = 0, False
                for s_id, s in enumerate(structure):
                    si = s[0] + i
                    sj = s[1] + j
                    sk = s[2] + k

                    s_ijk = segmentation[i, j, k]
                    ss_ijk = segmentation[si, sj, sk]

                    if s_ijk != ss_ijk:
                        label = cantor_sym_pair(s_ijk, ss_ijk)
                        is_full = True
                    else:
                        label = 0

                    struct_label_array[s_id] = label

                if is_full:
                    final_label, count = 0, 0
                    for label in struct_label_array:
                        temp_cont = 0
                        if label == final_label:
                            break

                        for test_label in struct_label_array:
                            if test_label == label:
                                temp_cont += 1

                        if temp_cont > count:
                            count = temp_cont
                            final_label = label

                    if count > min_counts:
                        out_edges[i, j, k] = final_label

    return out_edges


@njit(parallel=True)
def _create_rag_boundary_image(segmentation, structure, min_counts=0):
    
    shape = segmentation.shape
    out_edges = np.zeros((shape[0], shape[1], shape[2]), dtype='int64')
    label_dict = Dict.empty(key_type=types.int64, value_type=types.int64, )

    for i in prange(1, shape[0] - 1):
        for j in prange(1, shape[1] - 1):
            for k in prange(1, shape[2] - 1):
                is_full = False
                for s in structure:
                    si = s[0] + i
                    sj = s[1] + j
                    sk = s[2] + k
                    
                    s_ijk = segmentation[i, j, k]
                    ss_ijk = segmentation[si, sj, sk] 
                    
                    if s_ijk != ss_ijk:
                        if not is_full:
                            label_dict = Dict.empty(key_type=types.int64, value_type=types.int64,)
                            is_full = True
                            
                        label = cantor_sym_pair(s_ijk, ss_ijk)
                        label_dict[label] = label_dict.get(label, 1) + 1
                    
                if is_full and len(label_dict) > min_counts:
                    max_count, edge_label = 0, 0
                    for _key, _value in label_dict.items():
                        if _value > max_count:
                            edge_label, max_count = _key, _value
                            
                    out_edges[i, j, k] = edge_label
    return out_edges


@njit(parallel=True)
def _create_boundaries_image(segmentation, structure):
    shape = segmentation.shape
    out_edges = np.zeros((shape[0], shape[1], shape[2]), dtype='int32')

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


def create_boundaries_image(segmentation):
    return _create_boundaries_image(segmentation, get_neighborhood_structure())


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
        print("rag_boundaries is not a edges_ids subset")
        for value in edges_from_image.difference(edges_from_rag):
            edge_image[edge_image == value] = 0
            
    return edge_image


def create_rag_boundary_from_seg(segmentation, edges_ids=None, min_counts=0):

    rag_boundaries = _create_rag_boundary_image(segmentation, get_neighborhood_structure(), min_counts)
    if edges_ids is not None:
        rag_boundaries = rectify_edge_image(rag_boundaries, edges_ids)

    return rag_boundaries
