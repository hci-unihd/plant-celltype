import numpy as np
from skspatial.objects import Vector, Plane, Points

from plantcelltype.features.utils import check_valid_idx
from plantcelltype.utils.utils import cantor_sym_depair, create_edge_mapping, create_cell_mapping


def fit_plane_from_points(points_coo, cell_com):
    plane = Plane.best_fit(points_coo)
    central_point, vector = plane.point, plane.vector
    vector = - plane.side_point(cell_com) * vector
    return central_point, vector


def compute_edges_planes(cell_ids,
                         edges_ids,
                         cell_com,
                         edges_samples,
                         edges_com,
                         origin=(0, 0, 0),
                         min_valid_idx=20):
    """
    Find approximate edge planes
    """
    edge_sampling_mapping = create_edge_mapping(edges_ids, edges_samples)
    cell_com_mapping = create_cell_mapping(cell_ids, cell_com)
    edge_com_mapping = create_edge_mapping(edges_ids, edges_com)
    origin = np.array(origin)

    planes_planes = np.zeros((edges_ids.shape[0], 2, 3))
    for i, (edge_idx, points) in enumerate(edge_sampling_mapping.items()):
        # check if the ege is large enough to be sample
        valid_points, counts = check_valid_idx(points, origin)

        # center cell is required for orientation
        e1, e2 = cantor_sym_depair(edge_idx)
        center_cell = e1 if e1 != 0 else e2

        if counts > min_valid_idx and not Points(valid_points).are_collinear():
            # plane computation
            central_point, vector = fit_plane_from_points(valid_points, cell_com_mapping[center_cell])
        else:
            # if samples are not large enough find a proxy for the plane
            center_cell_com, central_point = cell_com_mapping[center_cell], edge_com_mapping[edge_idx]
            vector = Vector.from_points(center_cell_com, central_point)

            if vector.norm() < 1e-16:
                # random vector if the distance is close to zero
                vector = Vector(np.random.rand(3))
        planes_planes[i, 0] = central_point
        planes_planes[i, 1] = vector.unit()
    return planes_planes
