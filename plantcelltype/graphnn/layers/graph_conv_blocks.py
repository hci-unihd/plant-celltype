import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, BatchNorm, TransformerConv
from plantcelltype.graphnn.layers.activations import activations


class AbstractGCLayer(torch.nn.Module):
    def __init__(self,
                 graph_conv_module,
                 out_features,
                 activation='relu',
                 batch_norm=False,
                 batch_norm_feat=None,
                 drop_out=0.):
        super(AbstractGCLayer, self).__init__()
        assert activation in activations.keys()
        assert 0 <= drop_out <= 1

        self.g_conv = graph_conv_module
        self.activation = activations[activation]

        batch_norm_feat = out_features if batch_norm_feat is None else batch_norm_feat
        self.bn = BatchNorm(batch_norm_feat) if batch_norm else None
        self.dropout = torch.nn.Dropout(p=drop_out) if drop_out > 0. else None

    def forward(self, x, edge_index, **kwargs):
        x = self.g_conv(x, edge_index, **kwargs)
        x = x if self.bn is None else self.bn(x)
        x = self.activation(x)
        x = x if self.dropout is None else self.dropout(x)
        return x


class GCNLayer(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 normalize=True,
                 cached=False,
                 add_self_loops=True,
                 bias=True,
                 activation='relu',
                 batch_norm=False,
                 drop_out=0.):
        super(GCNLayer, self).__init__()
        assert type(in_features) is int and in_features > 0
        assert type(out_features) is int and out_features > 0

        g_module = GCNConv(in_features,
                           out_features,
                           normalize=normalize,
                           cached=cached,
                           add_self_loops=add_self_loops,
                           bias=bias,
                           )
        self.g_conv = AbstractGCLayer(g_module,
                                      out_features,
                                      activation=activation,
                                      batch_norm=batch_norm,
                                      drop_out=drop_out)

    def forward(self, x, edge_index):
        return self.g_conv(x, edge_index)


class GATLayer(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 heads=1,
                 gat_version='v1',
                 add_self_loops=True,
                 bias=True,
                 concat=True,
                 activation='relu',
                 batch_norm=False,
                 drop_out=0.):
        super(GATLayer, self).__init__()
        assert type(in_features) is int and in_features > 0
        assert type(out_features) is int and out_features > 0

        if gat_version == 'v1':
            g_module = GATConv(in_features,
                               out_features,
                               heads=heads,
                               concat=concat,
                               add_self_loops=add_self_loops,
                               bias=bias)
        elif gat_version == 'v2':
            g_module = GATv2Conv(in_features,
                                 out_features,
                                 heads=heads,
                                 concat=concat,
                                 add_self_loops=add_self_loops,
                                 bias=bias)
        else:
            raise NotImplemented

        batch_norm_feat = out_features * heads if concat and heads > 1 else None
        self.g_conv = AbstractGCLayer(g_module,
                                      out_features,
                                      activation=activation,
                                      batch_norm=batch_norm,
                                      batch_norm_feat=batch_norm_feat,
                                      drop_out=drop_out)

    def forward(self, x, edge_index):
        return self.g_conv(x, edge_index)


class TransformerGCNLayer(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 in_edges=None,
                 heads=1,
                 concat=True,
                 bias=True,
                 activation='relu',
                 batch_norm=False,
                 drop_out=0.):
        super(TransformerGCNLayer, self).__init__()
        assert type(in_features) is int and in_features > 0
        assert type(out_features) is int and out_features > 0

        g_module = TransformerConv(in_features,
                                   out_features,
                                   heads=heads,
                                   edge_dim=in_edges,
                                   concat=concat,
                                   bias=bias)

        batch_norm_feat = out_features * heads if concat and heads > 1 else None
        self.g_conv = AbstractGCLayer(g_module,
                                      out_features,
                                      activation=activation,
                                      batch_norm=batch_norm,
                                      batch_norm_feat=batch_norm_feat,
                                      drop_out=drop_out)

    def forward(self, x, edge_index, edge_attr=None):
        return self.g_conv(x, edge_index, edge_attr=edge_attr)
