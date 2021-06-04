import numpy as np
from skspatial.objects import Vector, Plane
from plantcelltype.utils.utils import cantor_sym_depair, create_edge_mapping, create_cell_mapping


def _valid_idx(samples, zeros=(0, 0, 0)):
    zeros = np.array(zeros)
    counts = 0
    valid_idx = []
    for i, _point in enumerate(samples):
        if not np.allclose(_point, zeros):
            counts += 1
            valid_idx.append(i)
    return valid_idx, counts


def _plane_from_points(points, cell_com):
    plane = Plane.best_fit(points)
    central_point, vector = plane.point, plane.vector
    vector = - plane.side_point(cell_com) * vector
    return central_point, vector


def compute_edges_planes(edges_samples,
                         edges_ids,
                         cell_ids,
                         cell_com,
                         edges_com,
                         cell_htbg,
                         origin=(0, 0, 0),
                         min_valid_idx=20):
    edge_sampling_mapping = create_edge_mapping(edges_samples, edges_ids)
    cell_com_mapping = create_cell_mapping(cell_com, cell_ids)
    cell_sp_mapping = create_cell_mapping(cell_htbg, cell_ids)
    edge_com_mapping = create_edge_mapping(edges_com, edges_ids)
    origin = np.array(origin)

    planes_points, planes_vectors = [], []
    for i, (edge_idx, points) in enumerate(edge_sampling_mapping.items()):
        valid_idx, counts = _valid_idx(points, origin)
        e1, e2 = cantor_sym_depair(edge_idx)
        e1_sd, e2_sd = cell_sp_mapping.get(e1, 0), cell_sp_mapping.get(e2, 0)
        center_cell = e1 if e1_sd > e2_sd else e2
        if counts > min_valid_idx:
            central_point, vector = _plane_from_points(points[valid_idx], cell_com_mapping[center_cell])
            planes_points.append(central_point)
            planes_vectors.append(vector)
        else:
            planes_points.append(edge_com_mapping[edge_idx])
            planes_vectors.append(Vector.from_points(cell_com_mapping[center_cell],
                                                     edge_com_mapping[edge_idx]).unit())

    return np.stack([planes_points, planes_vectors], axis=1)
