import pytorch_lightning as pl
import torch.nn.functional as F
from egmodels.graph_models import GCN2, GCN3, GAT2, GAT3, TGCN2, TGCN3
import torch

models_pool = {'GCN3': GCN3, 'GCN2': GCN2,
               'GAT3': GAT3, 'GAT2': GAT2,
               'TGCN3': TGCN3, 'TGCN2': TGCN2}


def load_model(name, model_kwargs=None):
    return models_pool[name](**model_kwargs)


class GraphClassification(pl.LightningModule):

    def __init__(self, model_name, model_kwargs):
        super(GraphClassification, self).__init__()
        self.net = load_model(model_name, model_kwargs)
        self.lr = 1e-3
        self.wd = 1e-5

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    def forward(self, data):
        return self.net(data)

    def training_step(self, batch, batch_idx):
        # to generalize
        batch = self.net(batch)
        logits = batch.out
        pred = logits.max(1)[1]
        loss = F.nll_loss(logits, batch.y)
        acc = pred.eq(batch.y).sum().item() / batch.y.shape[0]
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # to generalize
        val_batch = self.net(val_batch)
        logits = val_batch.out
        pred = logits.max(1)[1]
        loss = F.nll_loss(logits, val_batch.y)
        acc = pred.eq(val_batch.y).sum().item() / val_batch.y.shape[0]
        self.log('val_loss', loss)
        self.log('val_acc', acc)
