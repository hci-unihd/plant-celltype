import numpy as np
from skspatial.objects import Plane, Vector

from plantcelltype.features.utils import label2com
from plantcelltype.utils.utils import edges_ids2cantor_ids, create_cell_mapping


def edges2com(rag_boundaries, edges_ids):
    edges_ids = edges_ids2cantor_ids(edges_ids)
    return label2com(rag_boundaries, edges_ids)


def compute_edges_labels(cell_ids, edges_ids, cell_labels):
    edges_labels = np.zeros(edges_ids.shape[0])

    labels_mapping = create_cell_mapping(cell_ids, cell_labels)
    labels_mapping[0] = 0
    for i, (e1, e2) in enumerate(edges_ids):
        l1, l2 = labels_mapping[e1], labels_mapping[e2]
        edges_labels[i] = 1 if l1 == l2 else 0

    return edges_labels


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
