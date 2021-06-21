import glob
from plantcelltype.utils import create_h5
from plantcelltype.features.cell_vector_features import compute_local_reference_axis2_pair
from plantcelltype.utils.utils import create_cell_mapping, filter_bg_from_edges
from plantcelltype.features.rag import rectify_rag_names
from plantcelltype.utils.axis_transforms import AxisTransformer, scale_points
import numpy as np
import h5py
import napari
import torch
from torch_geometric.data.data import Data
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, TransformerConv  # noqa
from skspatial.objects import Plane, Line, Point, Vector
from torch_geometric.data import DataLoader
import os
from datetime import datetime


class Net1(torch.nn.Module):
    def __init__(self, in_feat=13, in_edges=9):
        super(Net1, self).__init__()
        self.conv1 = GCNConv(in_feat, 64, cached=False)
        self.conv2 = GCNConv(64, 64, cached=False)
        self.conv3 = GCNConv(64, 10,  cached=False)
        # GCNConv(32, 12, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, None
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        y = F.relu(self.conv2(x, edge_index, edge_weight))
        y = F.dropout(y, training=self.training)

        x = y + x

        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class Net2(torch.nn.Module):
    def __init__(self, in_feat=13, in_edges=9):
        super(Net2, self).__init__()
        self.conv1 = TransformerConv(in_feat, 64, heads=3, edge_dim=in_edges, concat=False)
        self.conv2 = TransformerConv(64, 64, heads=3, edge_dim=9, concat=False)
        self.conv3 = TransformerConv(64, 10, heads=1, edge_dim=9, concat=True)
        # GCNConv(32, 12, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        y = F.relu(self.conv2(x, edge_index, edge_weight))
        y = F.dropout(y, training=self.training)

        x = y + x

        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class Net3(torch.nn.Module):
    def __init__(self, in_feat=13, in_edges=9):
        super(Net3, self).__init__()
        self.conv1 = TransformerConv(in_feat, 64, heads=3, concat=False)
        self.conv2 = TransformerConv(64, 64, heads=3, concat=False)
        self.conv3 = TransformerConv(64, 10, heads=1, concat=True)
        # GCNConv(32, 12, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, None
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        y = F.relu(self.conv2(x, edge_index, edge_weight))
        y = F.dropout(y, training=self.training)

        x = y + x

        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)