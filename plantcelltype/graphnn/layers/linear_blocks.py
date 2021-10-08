import torch
from plantcelltype.graphnn.layers.activations import activations


class SimpleLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, activation='relu', bias=True, batch_norm=False, drop_out=0.):
        super(SimpleLayer, self).__init__()
        assert type(in_features) is int and in_features > 0
        assert type(out_features) is int and out_features > 0
        assert activation in activations.keys()
        assert 0 <= drop_out <= 1

        self.lin = torch.nn.Linear(in_features, out_features, bias=bias)
        self.bn = torch.nn.BatchNorm1d(out_features) if batch_norm else None
        self.activation = activations[activation]
        self.dropout = torch.nn.Dropout(p=drop_out) if drop_out > 0. else None

    def forward(self, x):
        x = self.lin(x)
        x = x if self.bn is None else self.bn(x)
        x = self.activation(x)
        x = x if self.dropout is None else self.dropout(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, features, layers_setup=None):
        super(MLP, self).__init__()
        if layers_setup is None:
            self.mlp = torch.nn.Sequential(*[SimpleLayer(in_features=features[i - 1],
                                                         out_features=features[i]) for i in range(1, len(features))])
        else:
            assert len(features) == len(layers_setup) + 1, f'length of features should be length of layer_setup + 1'
            self.mlp = torch.nn.Sequential(*[SimpleLayer(in_features=features[i - 1],
                                                         out_features=features[i],
                                                         **kwargs) for i, kwargs in enumerate(layers_setup, 1)])

    def forward(self, feat):
        return self.mlp(feat)
