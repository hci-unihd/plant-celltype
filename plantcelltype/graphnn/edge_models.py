import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse.tensor import SparseTensor
from plantcelltype.graphnn.layers.classifier import ClassifierMLP2
from plantcelltype.graphnn.line_graph import to_line_graph, mix_node_features
from plantcelltype.graphnn.layers.graph_conv_blocks import GCNLayer, TransformerGCNLayer


class LineGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(LineGCN2, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_feat, **{'batch_norm': True})

        self.gcn2 = GCNLayer(hidden_feat, hidden_feat, **{'batch_norm': True})

        self.gcn_line = GCNLayer(2*hidden_feat, out_features, **{'activation': 'none'})

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        line_x, line_edges_index = to_line_graph(x, edge_index, node_feat_mixing='cat')
        x = self.gcn_line(line_x, line_edges_index)
        return x


class EGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(EGCN2, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_feat, **{'batch_norm': True})

        self.gcn2 = GCNLayer(hidden_feat, hidden_feat, **{'batch_norm': True})

        self.mlp = ClassifierMLP2(2*hidden_feat, out_features)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        edges_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(edges_x)
        return x


class ETGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(ETGCN2, self).__init__()
        self.t_gcn1 = TransformerGCNLayer(in_features, hidden_feat, **{'batch_norm': True,
                                                                       'concat': True,
                                                                       'heads': 3})
        self.t_gcn2 = TransformerGCNLayer(3 * hidden_feat, hidden_feat, **{'batch_norm': True,
                                                                           'concat': True,
                                                                           'heads': 1
                                                                           })

        self.mlp = ClassifierMLP2(2 * hidden_feat, out_features)

    def forward(self, x, edge_index):
        x = self.t_gcn1(x, edge_index)
        x = self.t_gcn2(x, edge_index)
        line_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(line_x)
        return x


class LineTGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(LineTGCN2, self).__init__()
        self.t_gcn1 = TransformerGCNLayer(in_features, hidden_feat, **{'batch_norm': True,
                                                                       'concat': True,
                                                                       'heads': 3})
        self.t_gcn2 = TransformerGCNLayer(3 * hidden_feat, hidden_feat, **{'batch_norm': True,
                                                                           'concat': True,
                                                                           'heads': 1
                                                                           })
        self.lt_gcn1 = TransformerGCNLayer(2 * hidden_feat, out_features, **{'activation': 'none'})

    def forward(self, x, edge_index):
        x = self.t_gcn1(x, edge_index)
        x = self.t_gcn2(x, edge_index)
        line_x, line_edges_index = to_line_graph(x, edge_index, node_feat_mixing='cat')
        x = self.lt_gcn1(line_x, line_edges_index)
        return x


class EDeeperGCN(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat,
                 num_layers,
                 in_edges=None,
                 dropout=0.1):
        super(EDeeperGCN, self).__init__()

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

        self.mlp = ClassifierMLP2(2 * hidden_feat, out_features, hidden_feat=hidden_feat)

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

        line_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(line_x)
        return x


class EGCNII(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(EGCNII, self).__init__()

        self.lin = Linear(in_features, hidden_feat)

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_feat, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout
        self.mlp = ClassifierMLP2(2 * hidden_feat, out_features, hidden_feat=hidden_feat)

    def forward(self, x, edge_index):
        (row, col), values = gcn_norm(edge_index=edge_index)
        adj = SparseTensor(row=row, col=col, value=values)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lin(x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)

        line_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(line_x)
        return x


class LineEDeeperGCN(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat,
                 num_layers,
                 in_edges=None,
                 dropout=0.1):
        super(LineEDeeperGCN, self).__init__()

        self.node_encoder = Linear(in_features, hidden_feat)
        self.in_edges = in_edges
        if self.in_edges is not None:
            self.edge_encoder = Linear(in_edges, hidden_feat)
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        hidden_feat = 2 * hidden_feat
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_feat, hidden_feat, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_feat, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_feat, out_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)

        edge_attr = None
        x, edge_index = to_line_graph(x, edge_index, node_feat_mixing='cat')
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.lin(x)


class LineEGCNII(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(LineEGCNII, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_features, hidden_feat))
        self.lins.append(Linear(2 * hidden_feat, out_features))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(2 * hidden_feat, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        x, line_edges_index = to_line_graph(x, edge_index, node_feat_mixing='cat')

        (row, col), values = gcn_norm(edge_index=line_edges_index)
        adj = SparseTensor(row=row, col=col, value=values)

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
