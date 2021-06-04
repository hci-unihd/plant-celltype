from plantcelltype.utils.utils import create_edge_mapping, create_cell_mapping, cantor_sym_pair
from skspatial.objects import Vector
import numpy as np


def compute_local_reference_axis1(cell_ids, nx_graph, cell_htbg, edges_ids, edges_plane_vectors):

    cell_htbg_mapping = create_cell_mapping(cell_htbg, cell_ids)
    edges_pv_mapping = create_edge_mapping(edges_plane_vectors, edges_ids)
    cell_axis1_mapping = {}
    for c_idx in cell_ids:
        list_nx = list(nx_graph.neighbors(c_idx))
        c_idx_htbg = cell_htbg_mapping[c_idx]

        if c_idx_htbg == 1:
            # surface cells have only one direction
            cell_axis1_mapping[c_idx] = edges_pv_mapping[cantor_sym_pair(0, c_idx)][1]
        else:
            # not surface cells
            local_vectors = Vector([0.0, 0.0, 0.0])
            for cn_idx in list_nx:
                cn_idx_htbg = cell_htbg_mapping[cn_idx]
                if cn_idx_htbg < c_idx_htbg:
                    edges_pv = edges_pv_mapping[cantor_sym_pair(cn_idx, c_idx)]
                    local_vectors += Vector(edges_pv[1])
            local_vectors = local_vectors.unit()
            cell_axis1_mapping[c_idx] = local_vectors

    lr_axis_1 = np.array(list(cell_axis1_mapping.values()))
    return lr_axis_1


