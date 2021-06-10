from plantcelltype.utils.utils import create_edge_mapping, create_cell_mapping, cantor_sym_pair
from skspatial.objects import Vector, Line
from plantcelltype.features.rag import build_nx_graph
import numpy as np


def compute_local_reference_axis1(cell_ids, edges_ids, cell_hops_to_bg, edges_plane_vectors):

    nx_graph = build_nx_graph(cell_ids, edges_ids)
    cell_hops_to_bg_mapping = create_cell_mapping(cell_ids, cell_hops_to_bg)
    edges_pv_mapping = create_edge_mapping(edges_ids, edges_plane_vectors)
    cell_axis1_mapping = {}
    for c_idx in cell_ids:
        list_nx = list(nx_graph.neighbors(c_idx))
        c_idx_hops = cell_hops_to_bg_mapping[c_idx]

        if c_idx_hops == 1:
            # surface cells have only one direction
            cell_axis1_mapping[c_idx] = edges_pv_mapping[cantor_sym_pair(0, c_idx)][1]
        else:
            # not surface cells
            local_vectors = Vector([0.0, 0.0, 0.0])
            for cn_idx in list_nx:
                cn_idx_hops = cell_hops_to_bg_mapping[cn_idx]
                if cn_idx_hops < c_idx_hops:
                    edges_pv = edges_pv_mapping[cantor_sym_pair(cn_idx, c_idx)]
                    local_vectors += Vector(edges_pv[1])
            local_vectors = local_vectors.unit()
            cell_axis1_mapping[c_idx] = local_vectors

    lr_axis_1 = np.array(list(cell_axis1_mapping.values()))
    return lr_axis_1


def compute_local_reference_axis2_pair(cell_ids,
                                       edges_ids,
                                       cell_com,
                                       edges_com,
                                       global_axis=(1, 0, 0),
                                       weights=(1, 1)):
    global_axis = Vector(global_axis)
    nx_graph = build_nx_graph(cell_ids, edges_ids)
    cell_com_mapping = create_cell_mapping(cell_ids, cell_com)
    edges_com_mapping = create_edge_mapping(edges_ids, edges_com)
    min_points = []
    lr_axis_2 = []
    for n in cell_ids:
        min_dot = 1.
        com_n = cell_com_mapping[n]
        list_nx = list(nx_graph.neighbors(n))
        _min_points = [n, 0, -1]
        _vector = global_axis

        for i, a_n1 in enumerate(list_nx):
            com_n1 = cell_com_mapping[a_n1]
            cv1 = Vector.from_points(com_n, com_n1)

            com_e1 = edges_com_mapping[cantor_sym_pair(n, a_n1)]
            ev1 = Vector.from_points(com_n, com_e1)
            if ev1.norm() < 1e-16:
                ev1 = Vector(np.random.rand(3))

            for a_n2 in list_nx[i + 1:]:
                com_n2 = cell_com_mapping[a_n2]
                cv2 = Vector.from_points(com_n, com_n2)

                com_e2 = edges_com_mapping[cantor_sym_pair(n, a_n2)]
                ev2 = Vector.from_points(com_n, com_e2)
                if ev2.norm() < 1e-16:
                    ev2 = Vector(np.random.rand(3))

                cv_similarity = cv1.cosine_similarity(cv2)
                ev_similarity = ev1.cosine_similarity(ev2)
                tot_similarity = (weights[0] * cv_similarity + weights[1] * ev_similarity) / (weights[0] + weights[1])
                if tot_similarity < min_dot:
                    min_dot = tot_similarity
                    fv1 = cv1 + ev1
                    fv2 = cv2 + ev2
                    fv1_g_sym = fv1.cosine_similarity(global_axis)
                    fv2_g_sym = fv2.cosine_similarity(global_axis)

                    if fv1_g_sym > fv2_g_sym:
                        _vector = cv1.unit()
                        _min_points = [n, a_n1, tot_similarity]
                    else:
                        _vector = cv2.unit()
                        _min_points = [n, a_n2, tot_similarity]

        min_points.append(_min_points)
        lr_axis_2.append(_vector)

    return np.array(lr_axis_2), np.array(min_points)


def local_vectors_alignment(nx_graph, vectors_mapping):
    all_vectors = {}
    for n, current_vector in vectors_mapping.items():
        min_dot = 1.
        list_nx = list(nx_graph.neighbors(n))

        energy = 0
        for i, v_n in enumerate(list_nx):
            energy += current_vector.cosine_similarity(vectors_mapping[v_n])

        if energy > 0:
            all_vectors[n] = current_vector
        else:
            all_vectors[n] = - 1 * current_vector

    return

