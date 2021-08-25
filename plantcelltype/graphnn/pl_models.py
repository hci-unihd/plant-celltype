import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from egmodels.graph_base_models import GCN2, GCN3, GAT2, GAT3, TransformerGCN2, TransformerGCN3
from plantcelltype.graphnn.edge_models import LineGCN2, LineTGCN2, EGCN2, ETGCN2
from plantcelltype.graphnn.graph_models import GCNII, DeeperGCN

models_pool = {'GCN3': GCN3, 'GCN2': GCN2,
               'GAT3': GAT3, 'GAT2': GAT2,
               'GCNII': GCNII, 'DeeperGCN': DeeperGCN,
               'TransformerGCN3': TransformerGCN3,
               'TransformerGCN2': TransformerGCN2,
               'LineGCN2': LineGCN2,
               'LineTGCN2': LineTGCN2,
               'EGCN2': EGCN2,
               'ETGCN2': ETGCN2}


def load_model(name, model_kwargs=None):
    return models_pool[name](**model_kwargs)


class NodesClassification(pl.LightningModule):
    def __init__(self, config_model, config_optimizer):
        super(NodesClassification, self).__init__()
        self.colors = 255 * torch.randn(15, 3)
        self.net = load_model(config_model['name'], config_model['kwargs'])
        self.config_optimizer = config_optimizer

    def configure_optimizers(self):
        lr = float(self.config_optimizer['lr'])
        wd = float(self.config_optimizer['wd'])

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def forward(self, data):
        data = self.net(data)
        logits = data.out
        return data, logits

    def training_step(self, batch, batch_idx):
        # to generalize
        batch = self.net(batch)
        logits = torch.log_softmax(batch.out, 1)
        pred = logits.max(1)[1]
        loss = F.nll_loss(logits, batch.y)
        acc = pred.eq(batch.y).sum().item() / batch.y.shape[0]
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # to generalize
        val_batch = self.net(val_batch)
        logits = torch.log_softmax(val_batch.out, 1)
        pred = logits.max(1)[1]
        loss = F.nll_loss(logits, val_batch.y)
        correct_pred = pred.eq(val_batch.y)
        acc = correct_pred.sum().item() / val_batch.y.shape[0]

        self.log_meshes(val_batch.pos, pred, correct_pred, batch_idx)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def log_meshes(self, pos, pred, cor_pred, batch_idx):
        tensorboard = self.logger.experiment
        pos = torch.unsqueeze(pos, 0)

        color = torch.empty_like(pos)
        cor_pred = cor_pred.long()
        for i in range(color.shape[1]):
            color[0, i, :] = self.colors[cor_pred[i]]

        config_pc = {'material': {'cls': 'PointsMaterial', 'size': 5}}

        tensorboard.add_mesh(f'test_mesh_{batch_idx}', pos,
                             colors=color,
                             config_dict=config_pc, global_step=self.global_step)


class EdgesClassification(pl.LightningModule):
    def __init__(self, config_model, config_optimizer):
        super(EdgesClassification, self).__init__()
        print(config_model)
        self.net = load_model(config_model['name'], config_model['kwargs'])
        self.config_optimizer = config_optimizer

    def configure_optimizers(self):
        lr = float(self.config_optimizer['lr'])
        wd = float(self.config_optimizer['wd'])

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        return optimizer

    def forward(self, data):
        data = self.net(data)
        logits = data.out
        return data, logits

    def training_step(self, batch, batch_idx):
        # to generalize
        batch = self.net(batch)
        logits = batch.out[:, 0]
        logits = torch.sigmoid(logits)
        pred = logits > 0.5
        loss = F.binary_cross_entropy(logits, batch.edge_y.float())
        acc = pred.eq(batch.edge_y).sum().item() / batch.edge_y.shape[0]
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # to generalize
        val_batch = self.net(val_batch)
        logits = val_batch.out[:, 0]
        logits = torch.sigmoid(logits)
        pred = logits > 0.5
        loss = F.binary_cross_entropy(logits, val_batch.edge_y.float())
        acc = pred.eq(val_batch.edge_y).sum().item() / val_batch.edge_y.shape[0]
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
