from plantcelltype.features.utils import label2com
from plantcelltype.utils.utils import edges_ids2cantor_ids, create_cell_mapping
from skspatial.objects import Plane, Vector
import numpy as np


def edges2com(rag_boundaries, edges_ids):
    edges_ids = edges_ids2cantor_ids(edges_ids)
    return label2com(rag_boundaries, edges_ids)


def compute_boundaries_com_angles(cell_ids, edges_ids, cell_com, edges_sample):
    cell_com_mapping = create_cell_mapping(cell_ids, cell_com)

    angles = np.zeros(edges_ids.shape[0])
    for i, (samples, (e1, e2)) in enumerate(zip(edges_sample, edges_ids)):
        if e1 > 0 and e2 > 0:
            try:
                plane = Plane.best_fit(samples)
                vector = Vector.from_points(cell_com_mapping[e1], cell_com_mapping[e2])
                projection = plane.project_vector(vector)
                _angle = projection.unit().dot(vector.unit())
                angles[i] = _angle
            except:
                angles[i] = 0

        else:
            angles[i] = 0
    return angles


def compute_edges_length(cell_ids, edges_ids, cell_com):
    com_mapping = create_cell_mapping(cell_ids, cell_com)
    com_distance = np.zeros(edges_ids.shape[0])
    for i, (e1, e2) in enumerate(edges_ids):
        if e1 > 0 and e2 > 0:
            x1, x2 = com_mapping[e1], com_mapping[e2]
            d = np.sqrt(np.sum((x1 - x2) ** 2))
            com_distance[i] = d
        else:
            com_distance[i] = 0

    return com_distance


def compute_edges_angles(cell_ids, edges_ids, cell_com, edges_com):
    com_mapping = create_cell_mapping(cell_ids, cell_com)
    com_angle = np.zeros(edges_ids.shape[0])
    for i, (e1, e2) in enumerate(edges_ids):
        if e1 > 0 and e2 > 0:
            try:
                x1, x2 = com_mapping[e1], com_mapping[e2]
                e_com = edges_com[i]
                vector1 = Vector.from_points(x1, e_com).unit()
                vector2 = Vector.from_points(x2, e_com).unit()
                _angle = vector1.dot(vector2)
                com_angle[i] = abs(_angle)
            except:
                com_angle[i] = 0
        else:
            com_angle[i] = 0

    return com_angle
