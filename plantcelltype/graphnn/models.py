import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv
import pytorch_lightning as pl


class TGCN(torch.nn.Module):
    def __init__(self, in_feat=13, out_feat=10, feat_layers=(64, 64), in_edges=9):
        super(TGCN, self).__init__()
        self.conv1 = TransformerConv(in_feat, 64, heads=3, edge_dim=in_edges, concat=False)
        self.conv2 = TransformerConv(64, 64, heads=3, edge_dim=in_edges, concat=False)
        self.conv3 = TransformerConv(64, out_feat, heads=1, edge_dim=in_edges, concat=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)

        y = F.relu(self.conv2(x, edge_index, edge_weight))
        y = F.dropout(y, training=self.training)

        x = y + x

        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
