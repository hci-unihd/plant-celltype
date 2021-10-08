import numpy as np
import torch
from plantcelltype.utils import create_cell_mapping


def concatenate_feat(f1, f2, axis=0):
    return torch.cat([f1, f2], axis)


def sum_feat(f1, f2, axis=0):
    return (f1 + f2)/2


def min_feat(f1, f2, axis=0):
    _cat_f = torch.stack([f1, f2], 0)
    return torch.min(_cat_f, 0)[0]


def max_feat(f1, f2, axis=0):
    _cat_f = torch.cat([f1, f2], 0)
    return torch.max(_cat_f, 0)[0]


def l2(f1, f2, axis=0):
    _l2 = (f1 - f2)**2
    return torch.sqrt(_l2)


_feat_mix = {'cat': concatenate_feat,
             'sum': sum_feat,
             'min': min_feat,
             'max': max_feat,
             'l2': l2}


def mix_node_features(x,
                      edge_index,
                      edges_attr=None,
                      node_feat_mixing='sum',
                      edges_feat_mixing='cat'):

    np_node_index = np.arange(x.shape[0])
    np_edge_index = edge_index.cpu().data.numpy().T

    node_feat_mapping = create_cell_mapping(np_node_index, x, safe_cast=False)

    line_node_feat = []
    for e1, e2 in np_edge_index:
        feat = [node_feat_mapping[e1], node_feat_mapping[e2]]
        line_node_feat.append(_feat_mix[node_feat_mixing](*feat, axis=0))

    line_node_feat = torch.stack(line_node_feat, 0)
    if edges_attr is not None:
        line_node_feat = _feat_mix[edges_feat_mixing](*(line_node_feat, edges_attr), axis=1)

    return line_node_feat, np_node_index, np_edge_index
