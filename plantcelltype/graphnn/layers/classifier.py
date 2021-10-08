import torch
from plantcelltype.graphnn.layers.linear_blocks import MLP


class ClassifierMLP2(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat=256):
        super(ClassifierMLP2, self).__init__()
        self.mlp = MLP([in_features, hidden_feat, out_features],
                       [{'batch_norm': True, 'activation': 'relu'},
                        {'activation': 'none'}])

    def forward(self, data):
        feat = data.feat
        out = self.mlp(feat)
        data.out = out
        return data


class ClassifierMLP3(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_feat_1=512, hidden_feat_2=256):
        super(ClassifierMLP3, self).__init__()
        self.mlp = MLP([in_features, hidden_feat_1, hidden_feat_2, out_features],
                       [{'batch_norm': True, 'activation': 'relu'},
                        {'batch_norm': True, 'activation': 'relu'},
                        {'activation': 'none'}])

    def forward(self, data):
        feat = data.feat
        out = self.mlp(feat)
        data.out = out
        return data
