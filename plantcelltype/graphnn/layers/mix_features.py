import numpy as np
import torch
from plantcelltype.utils import create_cell_mapping


def concatenate_feat(f1, f2, axis=0):
    return torch.cat([f1, f2], axis)


def sum_feat(f1, f2, axis=0):
    return f1 + f2


def mean_feat(f1, f2, axis=0):
    return sum_feat(f1, f2) / 2


def max_feat(f1, f2, axis=0):
    _cat_f = torch.cat([f1, f2], 0)
    return torch.max(_cat_f, 0)[0]


def l2(f1, f2, axis=0):
    _l2 = (f1 - f2) ** 2
    return torch.sqrt(_l2)


_feat_mix = {'cat': concatenate_feat,
             'sum': sum_feat,
             'mean': mean_feat,
             'max': max_feat,
             'l2': l2}


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
