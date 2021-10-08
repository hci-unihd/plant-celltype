import torch

from plantcelltype.graphnn.graph_models import GCN2, GAT2
from plantcelltype.graphnn.graph_models import GCNII, DeeperGCN
from plantcelltype.graphnn.layers.classifier import ClassifierMLP2
from plantcelltype.graphnn.layers.mix_features import mix_node_features


class AbstractEGCN(torch.nn.Module):
    def __init__(self, model,
                 in_features,
                 out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(AbstractEGCN, self).__init__()
        self.model = model(in_features=in_features,
                           out_features=hidden_feat,
                           hidden_feat=hidden_feat,
                           layer1_kwargs=layer1_kwargs,
                           layer2_kwargs=layer2_kwargs)

        self.mlp = ClassifierMLP2(2*hidden_feat, out_features)

    def forward(self, data):
        data = self.gcn(data)

        x, edge_index = data.out, data.edge_index
        edges_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(edges_x)
        data.out = x
        return data


class EGCN2(torch.nn.Module):
    def __init__(self, in_features,
                 out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(EGCN2, self).__init__()
        self.egcn = AbstractEGCN(GCN2,
                                 in_features=in_features,
                                 out_features=out_features,
                                 hidden_feat=hidden_feat,
                                 layer1_kwargs=layer1_kwargs,
                                 layer2_kwargs=layer2_kwargs)

    def forward(self, data):
        data = self.egcn(data)
        return data


class EGAT2(torch.nn.Module):
    def __init__(self, in_features,
                 out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(EGAT2, self).__init__()
        self.egat = AbstractEGCN(GAT2,
                                 in_features=in_features,
                                 out_features=out_features,
                                 hidden_feat=hidden_feat,
                                 layer1_kwargs=layer1_kwargs,
                                 layer2_kwargs=layer2_kwargs)

    def forward(self, data):
        data = self.egat(data)
        return data


class ETransformerGCN2(torch.nn.Module):
    def __init__(self, in_features,
                 out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(ETransformerGCN2, self).__init__()
        self.etgcn = AbstractEGCN(ETransformerGCN2,
                                  in_features=in_features,
                                  out_features=out_features,
                                  hidden_feat=hidden_feat,
                                  layer1_kwargs=layer1_kwargs,
                                  layer2_kwargs=layer2_kwargs)

    def forward(self, data):
        data = self.etgcn(data)
        return data


class EDeeperGCN(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat,
                 num_layers,
                 in_edges=None,
                 dropout=0.1):
        super(EDeeperGCN, self).__init__()
        self.deeper_gcn = DeeperGCN(in_features,
                                    hidden_feat,
                                    hidden_feat=hidden_feat,
                                    num_layers=num_layers,
                                    in_edges=in_edges,
                                    dropout=dropout)
        self.mlp = ClassifierMLP2(2 * hidden_feat,
                                  out_features,
                                  hidden_feat=hidden_feat)

    def forward(self, data):
        data = self.deeper_gcn(data)

        x, edge_index = data.out, data.edge_index
        edges_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(edges_x)
        data.out = x
        return data


class EGCNII(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(EGCNII, self).__init__()
        self.gcnii = GCNII(in_features,
                           hidden_feat,
                           hidden_feat=hidden_feat,
                           num_layers=num_layers,
                           alpha=alpha,
                           theta=theta,
                           shared_weights=shared_weights,
                           dropout=dropout)
        self.mlp = ClassifierMLP2(2 * hidden_feat, out_features, hidden_feat=hidden_feat)

    def forward(self, data):
        data = self.gcnii(data)

        x, edge_index = data.out, data.edge_index
        edges_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.mlp(edges_x)
        data.out = x
        return data
