import numpy as np
from skspatial.objects import Vector, Plane

from plantcelltype.features.utils import check_valid_idx
from plantcelltype.utils.utils import cantor_sym_depair, create_edge_mapping, create_cell_mapping


def _plane_from_points(points_coo, cell_com):
    plane = Plane.best_fit(points_coo)
    central_point, vector = plane.point, plane.vector
    vector = - plane.side_point(cell_com) * vector
    return central_point, vector


def compute_edges_planes(cell_ids,
                         edges_ids,
                         cell_com,
                         cell_hops_to_bg,
                         edges_samples,
                         edges_com,
                         origin=(0, 0, 0),
                         min_valid_idx=20):
    edge_sampling_mapping = create_edge_mapping(edges_ids, edges_samples)
    cell_com_mapping = create_cell_mapping(cell_ids, cell_com)
    cell_sp_mapping = create_cell_mapping(cell_ids, cell_hops_to_bg)
    edge_com_mapping = create_edge_mapping(edges_ids, edges_com)
    origin = np.array(origin)

    planes_points, planes_vectors = [], []
    for i, (edge_idx, points) in enumerate(edge_sampling_mapping.items()):
        valid_idx, counts = check_valid_idx(points, origin)
        e1, e2 = cantor_sym_depair(edge_idx)
        e1_sd, e2_sd = cell_sp_mapping.get(e1, 0), cell_sp_mapping.get(e2, 0)
        center_cell = e1 if e1_sd > e2_sd else e2
        if counts > min_valid_idx:
            central_point, vector = _plane_from_points(points[valid_idx], cell_com_mapping[center_cell])
            planes_points.append(central_point)
            planes_vectors.append(vector)
        else:
            center_cell_com, edges_idx_com = cell_com_mapping[center_cell], edge_com_mapping[edge_idx]
            planes_points.append(edges_idx_com)
            if not np.allclose(center_cell_com, edges_idx_com):
                planes_vectors.append(Vector.from_points(center_cell_com,
                                                         edges_idx_com).unit())

            else:
                planes_vectors.append(Vector(np.random.rand(3)).unit())

    return np.stack([planes_points, planes_vectors], axis=1)
