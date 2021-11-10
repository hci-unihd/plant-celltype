import numpy as np
import torch
from plantcelltype.utils import create_cell_mapping


def concatenate_feat(f1, f2, axis=0):
    return torch.cat([f1, f2], axis)


def sum_feat(f1, f2, axis=0):
    return (f1 + f2) / 2


def min_feat(f1, f2, axis=0):
    _cat_f = torch.stack([f1, f2], 0)
    return torch.min(_cat_f, 0)[0]


def max_feat(f1, f2, axis=0):
    _cat_f = torch.cat([f1, f2], 0)
    return torch.max(_cat_f, 0)[0]


def l2(f1, f2, axis=0):
    _l2 = (f1 - f2) ** 2
    return torch.sqrt(_l2)


_feat_mix = {'cat': concatenate_feat,
             'sum': sum_feat,
             'min': min_feat,
             'max': max_feat,
             'l2': l2}


def mix_node_features_safe(x,
                           edge_index,
                           edges_attr=None,
                           node_feat_mixing='sum',
                           edges_feat_mixing='cat'):
    np_node_index = np.arange(x.shape[0])
    np_edge_index = edge_index.cpu().data.numpy().T

    node_feat_mapping = create_cell_mapping(np_node_index, x, safe_cast=False)

    edge_feat = []
    for e1, e2 in np_edge_index:
        feat = [node_feat_mapping[e1], node_feat_mapping[e2]]
        edge_feat.append(_feat_mix[node_feat_mixing](*feat, axis=0))

    edge_feat = torch.stack(edge_feat, 0)
    if edges_attr is not None:
        edge_feat = _feat_mix[edges_feat_mixing](*(edge_feat, edges_attr), axis=1)

    return edge_feat, np_node_index, np_edge_index


def mix_node_features(node_feat,
                      edge_index,
                      edges_attr=None,
                      node_feat_mixing='sum',
                      edges_feat_mixing='cat'):
    edge_node_0, edge_node_1 = node_feat[edge_index[0]], node_feat[edge_index[1]]
    edge_feat = _feat_mix[node_feat_mixing](edge_node_0, edge_node_1, axis=1)

    if edges_attr is not None:
        edge_feat = _feat_mix[edges_feat_mixing](edge_feat, edges_attr, axis=1)

    return edge_feat
