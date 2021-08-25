import torch

from egmodels import tg_dispatch
from egmodels.classifier import ClassifierMLP2
from egmodels.layers.graph_conv_blocks import GCNLayer, TransformerGCNLayer
from plantcelltype.graphnn.line_graph import to_line_graph, mix_node_features


class LineGCN2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(LineGCN2, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_feat, **{'batch_norm': True})

        self.gcn2 = GCNLayer(hidden_feat, hidden_feat, **{'batch_norm': True})

        self.gcn_line = GCNLayer(2*hidden_feat, out_features, **{'activation': 'none'})

    @tg_dispatch()
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

    @tg_dispatch()
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

    @tg_dispatch()
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

    @tg_dispatch()
    def forward(self, x, edge_index):
        x = self.t_gcn1(x, edge_index)
        x = self.t_gcn2(x, edge_index)
        line_x, line_edges_index = to_line_graph(x, edge_index, node_feat_mixing='cat')
        x = self.lt_gcn1(line_x, line_edges_index)
        return x

