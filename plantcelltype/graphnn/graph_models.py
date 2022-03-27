import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from plantcelltype.graphnn.layers.graph_conv_blocks import GCNLayer, GATLayer, TransformerGCNLayer
from torch_sparse.tensor import SparseTensor
from torch_geometric.nn.models import GCN, GAT, GIN, GraphSAGE


class GenericTgModel(torch.nn.Module):
    def __init__(self, model, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):

        super(GenericTgModel, self).__init__()
        self.module = model(in_channels=in_features,
                            hidden_channels=hidden_feat,
                            num_layers=num_layers,
                            out_channels=out_features,
                            dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.module(x, edge_index)
        data.out = x
        return data


class TgGCN(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GCN,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class TgGAT(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GAT,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class TgGIN(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GIN,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class TgGraphSAGE(GenericTgModel):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0.,
                 ):
        super().__init__(GraphSAGE,
                         in_features=in_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         out_features=out_features,
                         dropout=dropout)


class GCN2(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(GCN2, self).__init__()

        _layer1_kwargs = {'batch_norm': True}
        _layer2_kwargs = {'activation': 'none'}

        if layer1_kwargs is not None:
            _layer1_kwargs.update(layer1_kwargs)

        if layer2_kwargs is not None:
            _layer2_kwargs.update(layer2_kwargs)

        self.gcn1 = GCNLayer(in_features, hidden_feat, **_layer1_kwargs)
        self.gcn2 = GCNLayer(hidden_feat, out_features, **_layer2_kwargs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        data.out = x
        return data


class GAT2(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(GAT2, self).__init__()

        _layer1_kwargs = {'batch_norm': True, 'concat': True, 'heads': 3}
        _layer2_kwargs = {'activation': 'none', 'concat': True, 'heads': 1}

        if layer1_kwargs is not None:
            _layer1_kwargs.update(layer1_kwargs)

        if layer2_kwargs is not None:
            _layer2_kwargs.update(layer2_kwargs)

        self.gat1 = GATLayer(in_features, hidden_feat, **_layer1_kwargs)

        in_features_2 = _layer1_kwargs['heads'] if _layer1_kwargs['concat'] and _layer1_kwargs['heads'] > 1 else 1
        in_features_2 *= hidden_feat
        self.gat2 = GATLayer(in_features_2, out_features, **_layer2_kwargs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        data.out = x
        return data


class GAT2v2(GAT2):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         hidden_feat=hidden_feat,
                         layer1_kwargs=layer1_kwargs,
                         layer2_kwargs=layer2_kwargs
                         )


class TransformerGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, in_edges=None,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(TransformerGCN2, self).__init__()

        _layer1_kwargs = {'batch_norm': True, 'concat': True, 'heads': 3}
        _layer2_kwargs = {'activation': 'none', 'concat': True, 'heads': 1}

        if layer1_kwargs is not None:
            _layer1_kwargs.update(layer1_kwargs)

        if layer2_kwargs is not None:
            _layer2_kwargs.update(layer2_kwargs)

        self.in_edges = in_edges
        self.t_gcn1 = TransformerGCNLayer(in_features, hidden_feat, in_edges, **_layer1_kwargs)

        in_features_2 = _layer1_kwargs['heads'] if _layer1_kwargs['concat'] and _layer1_kwargs['heads'] > 1 else 1
        in_features_2 *= hidden_feat
        self.t_gcn2 = TransformerGCNLayer(in_features_2, out_features, in_edges, **_layer2_kwargs)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr = None if self.in_edges is None else edge_attr
        x = self.t_gcn1(x, edge_index, edge_attr=edge_attr)
        x = self.t_gcn2(x, edge_index, edge_attr=edge_attr)
        data.out = x
        return data


class NoEdgesTransformerGCN2(TransformerGCN2):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         in_edges=None,
                         hidden_feat=hidden_feat,
                         layer1_kwargs=layer1_kwargs,
                         layer2_kwargs=layer2_kwargs
                         )


class DeeperGCN(torch.nn.Module):
    """
    Implementation adapted from:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py
    Credits to Matthias Fey
    """
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

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
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
        x = self.lin(x)
        data.out = x
        return data


class NoEdgesDeeperGCN(DeeperGCN):
    def __init__(self, in_features, out_features,
                 hidden_feat,
                 num_layers,
                 dropout=0.1):
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         hidden_feat=hidden_feat,
                         num_layers=num_layers,
                         in_edges=None,
                         dropout=dropout)


class GCNII(torch.nn.Module):
    """
    Implementation adapted from:
    https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn2_cora.py
    Credits to Matthias Fey
    """
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
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
        data.out = x
        return data
