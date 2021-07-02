import torch
from egmodels.layers.graph_conv_blocks import GCNLayer, TGCNLayer
from egmodels import tg_dispatch
from egmodels.classifier import ClassifierMLP2
from plantcelltype.graphnn.line_graph import to_line_graph, mix_node_features


class LineGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(LineGCN2, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_feat, **{'batch_norm': True,
                                                           'activation': 'relu',
                                                           'add_self_loops': True})

        self.gcn2 = GCNLayer(hidden_feat, hidden_feat, **{'batch_norm': False,
                                                           'activation': 'relu',
                                                           'add_self_loops': True})

        self.gcn_line = GCNLayer(2*hidden_feat, out_features, **{'batch_norm': False,
                                                           'activation': 'none',
                                                           'add_self_loops': True})

    @tg_dispatch()
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        line_x, line_edges_index = to_line_graph(x, edge_index, node_feat_mixing='cat')
        x = self.gcn_line(line_x, line_edges_index)
        x = torch.sigmoid(x)
        return x


class EGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(EGCN2, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_feat, **{'batch_norm': True,
                                                           'activation': 'relu',
                                                           'add_self_loops': True})

        self.gcn2 = GCNLayer(hidden_feat, hidden_feat, **{'batch_norm': False,
                                                           'activation': 'relu',
                                                           'add_self_loops': True})

        self.mlp = ClassifierMLP2(2*hidden_feat, out_features)

    @tg_dispatch()
    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        edges_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(edges_x)
        x = torch.sigmoid(x)
        return x


class ETGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(ETGCN2, self).__init__()
        self.t_gcn1 = TGCNLayer(in_features, hidden_feat, **{'batch_norm': True,
                                                             'activation': 'relu'})
        self.t_gcn2 = TGCNLayer(hidden_feat, hidden_feat, **{'batch_norm': False,
                                                             'activation': 'relu'})

        self.mlp = ClassifierMLP2(2 * hidden_feat, out_features)

    @tg_dispatch()
    def forward(self, x, edge_index):
        x = self.t_gcn1(x, edge_index)
        x = self.t_gcn2(x, edge_index)
        line_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(line_x)
        x = torch.sigmoid(x)
        return x


class LineTGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(LineTGCN2, self).__init__()
        self.t_gcn1 = TGCNLayer(in_features, hidden_feat, **{'batch_norm': True,
                                                             'activation': 'relu'})
        self.t_gcn2 = TGCNLayer(hidden_feat, hidden_feat, **{'batch_norm': False,
                                                             'activation': 'relu'})

        self.lt_gcn1 = TGCNLayer(2 * hidden_feat, out_features, **{'batch_norm': False,
                                                                   'activation': 'none'})

    @tg_dispatch()
    def forward(self, x, edge_index):
        x = self.t_gcn1(x, edge_index)
        x = self.t_gcn2(x, edge_index)
        line_x, line_edges_index = to_line_graph(x, edge_index, node_feat_mixing='cat')
        x = self.lt_gcn1(line_x, line_edges_index)
        x = torch.sigmoid(x)
        return x

