import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse.tensor import SparseTensor

from egmodels import tg_dispatch


class DeeperGCN(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat,
                 num_layers,
                 in_edges=None,
                 dropout=0.1):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(in_features, hidden_feat)
        self.in_edges = in_edges
        if self.in_edges is not None:
            self.edge_encoder = Linear(in_edges, hidden_feat)
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_feat, hidden_feat, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_feat, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_feat, out_features)

    @tg_dispatch()
    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        if self.in_edges is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin(x)


class GCNII(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(GCNII, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_features, hidden_feat))
        self.lins.append(Linear(hidden_feat, out_features))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_feat, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    @tg_dispatch()
    def forward(self, x, edge_index):
        (row, col), values = gcn_norm(edge_index=edge_index)
        adj = SparseTensor(row=row, col=col, value=values)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
