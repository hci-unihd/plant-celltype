import torch

from plantcelltype.graphnn.graph_models import TgGCN, GAT2, TransformerGCN2
from plantcelltype.graphnn.graph_models import GCNII, DeeperGCN
from plantcelltype.graphnn.layers.mix_features import mix_node_features
from torch.nn import Linear


class AbstractEGCN(torch.nn.Module):
    def __init__(self, model,
                 in_features, out_features,
                 hidden_feat,
                 model_kwargs):
        super(AbstractEGCN, self).__init__()
        self.model = model(in_features,
                           out_features=hidden_feat,
                           hidden_feat=hidden_feat,
                           **model_kwargs
                           )

        self.lin = Linear(2 * hidden_feat, out_features)

    def forward(self, data):
        data = self.model(data)

        x, edge_index = data.out, data.edge_index
        edges_x, _, _ = mix_node_features(x, edge_index, node_feat_mixing='cat')
        x = self.lin(edges_x)
        data.out = x
        return data


class ETgGCN(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat=256,
                 num_layers=2,
                 dropout=0., ):
        super(ETgGCN, self).__init__()
        model_kwarg = {'num_layers': num_layers, 'dropout': dropout}
        self.e_gcn = AbstractEGCN(TgGCN,
                                  in_features, out_features,
                                  hidden_feat=hidden_feat,
                                  model_kwargs=model_kwarg)

    def forward(self, data):
        data = self.e_gcn(data)
        return data


class EGAT2(torch.nn.Module):
    def __init__(self, in_features,
                 out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(EGAT2, self).__init__()
        model_kwarg = {'layer1_kwargs': layer1_kwargs,
                       'layer2_kwargs': layer2_kwargs}
        self.e_gat = AbstractEGCN(GAT2,
                                  in_features=in_features,
                                  out_features=out_features,
                                  hidden_feat=hidden_feat,
                                  model_kwargs=model_kwarg)

    def forward(self, data):
        data = self.e_gat(data)
        return data


class ETransformerGCN2(torch.nn.Module):
    def __init__(self, in_features,
                 out_features,
                 hidden_feat=256,
                 layer1_kwargs=None,
                 layer2_kwargs=None):
        super(ETransformerGCN2, self).__init__()
        model_kwarg = {'layer1_kwargs': layer1_kwargs,
                       'layer2_kwargs': layer2_kwargs}
        self.e_tgcn = AbstractEGCN(TransformerGCN2,
                                   in_features=in_features,
                                   out_features=out_features,
                                   hidden_feat=hidden_feat,
                                   model_kwargs=model_kwarg)

    def forward(self, data):
        data = self.e_tgcn(data)
        return data


class EDeeperGCN(torch.nn.Module):
    def __init__(self, in_features, out_features,
                 hidden_feat,
                 num_layers,
                 in_edges=None,
                 dropout=0.1):
        super(EDeeperGCN, self).__init__()

        model_kwarg = {'num_layers': num_layers,
                       'in_edges': in_edges,
                       'dropout': dropout}
        self.e_deepergcn = AbstractEGCN(DeeperGCN,
                                        in_features=in_features,
                                        hidden_feat=hidden_feat,
                                        out_features=out_features,
                                        model_kwargs=model_kwarg)

    def forward(self, data):
        data = self.e_deepergcn(data)
        return data


class ENoEdgesDeeperGCN(EDeeperGCN):
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


class EGCNII(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(EGCNII, self).__init__()
        model_kwarg = {'num_layers': num_layers,
                       'alpha': alpha,
                       'theta': theta,
                       'shared_weights': shared_weights,
                       'dropout': dropout}

        self.e_gcnii = AbstractEGCN(GCNII,
                                    in_features=in_features,
                                    hidden_feat=hidden_feat,
                                    out_features=out_features,
                                    model_kwargs=model_kwarg)

    def forward(self, data):
        data = self.e_gcnii(data)
        return data
