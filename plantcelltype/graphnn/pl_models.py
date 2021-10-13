import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1

from pctg_benchmark.evaluation.metrics import NodeClassificationMetrics
from plantcelltype.graphnn.graph_models import GCN2, GCN3
from plantcelltype.graphnn.graph_models import GAT2, GAT3
from plantcelltype.graphnn.graph_models import TransformerGCN2, TransformerGCN3
from plantcelltype.graphnn.graph_models import GCNII, DeeperGCN
from plantcelltype.graphnn.edge_models import EGCN2, EGAT2, ETransformerGCN2
from plantcelltype.graphnn.edge_models import EDeeperGCN, EGCNII

models_pool = {'GCN3': GCN3, 'GCN2': GCN2,
               'GAT3': GAT3, 'GAT2': GAT2,
               'GCNII': GCNII, 'DeeperGCN': DeeperGCN,
               'TransformerGCN3': TransformerGCN3,
               'TransformerGCN2': TransformerGCN2,
               'EGCN2': EGCN2,
               'EGAT2': EGAT2,
               'ETransformerGCN2': ETransformerGCN2,
               'EDeeperGCN': EDeeperGCN,
               'EGCNII': EGCNII}


def load_model(name, model_kwargs=None):
    return models_pool[name](**model_kwargs)


class NodesClassification(pl.LightningModule):
    def __init__(self, model, optimizer, logger=None):
        super(NodesClassification, self).__init__()
        self.net = load_model(model['name'], model['kwargs'])
        self.optimizer = {} if optimizer is None else optimizer

        logger = {} if logger is None else logger
        self.log_points = logger.get('log_points', False)

        self.reference_metric, reference_mode, reference_default = 'accuracy_micro', 'max', 0
        self.saved_metrics = {'val': {'loss': {'value': 1e16, 'step': 0, 'mode': 'min'},
                                      self.reference_metric: {'value': reference_default,
                                                              'step': 0,
                                                              'mode': reference_mode},
                                      'results': [],
                                      'results_last': []},
                              'train': {'loss': {'value': 1e16, 'step': 0, 'mode': 'min'},
                                        self.reference_metric: {'value': reference_default,
                                                                'step': 0,
                                                                'mode': reference_mode}}
                              }

        out_class = model['kwargs']['out_features']
        self.classification_evaluation = NodeClassificationMetrics(out_class)
        self.micro_accuracy = Accuracy(average='micro')
        self.macro_accuracy = Accuracy(num_classes=out_class, average='macro')
        self.save_hyperparameters()

    def configure_optimizers(self):
        lr = float(self.optimizer.get('lr', 1e-5))
        wd = float(self.optimizer.get('wd', 1e-6))

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
        glob_acc = self.micro_accuracy(pred, batch.y)
        class_acc = self.macro_accuracy(pred, batch.y)

        self.log('train_loss', loss)
        self.log('train_global_acc', glob_acc)
        self.log('train_class_acc', class_acc)
        return loss

    def _val_forward(self, val_batch):
        # to generalize
        val_batch = self.net(val_batch)
        logits = torch.log_softmax(val_batch.out, 1)
        pred = logits.max(1)[1]
        loss = F.nll_loss(logits, val_batch.y)
        return pred, loss

    def validation_step(self, val_batch, batch_idx):
        pred, loss = self._val_forward(val_batch)
        full_metrics = self.classification_evaluation.compute_metrics(pred.cpu(),
                                                                      val_batch.y.cpu(),
                                                                      self.global_step)

        if self.log_points:
            self._log_points(val_batch.pos, pred, batch_idx)

        self.log('val_loss', loss)
        self.log('val_global_acc', full_metrics['accuracy_micro'])
        self.log('val_class_acc', full_metrics['accuracy_macro'])

        metrics = {'hp_metric': full_metrics['accuracy_micro']}
        self.log_dict(metrics)
        return full_metrics, val_batch.file_path

    def validation_epoch_end(self, outputs):
        epoch_acc = [val_sample[0][self.reference_metric] for val_sample in outputs]
        epoch_acc = sum(epoch_acc) / len(outputs)
        mode = self.saved_metrics['val'][self.reference_metric]['mode']
        check = self.saved_metrics['val'][self.reference_metric]['value'] - epoch_acc
        keys = ['results', 'file_path', 'meta']
        results = []
        for val_sample in outputs:
            _results = {key: value for key, value in zip(keys, val_sample)}
            _results.get('step', self.global_step)
            results.append(_results)

        if (mode == 'min' and check > 0) or (mode == 'max' and check < 0):
            self.saved_metrics['val']['results'] = results
            self.saved_metrics['val'][self.reference_metric]['value'] = epoch_acc
            self.saved_metrics['val'][self.reference_metric]['step'] = self.global_step
        self.saved_metrics['val']['results_last'] = results
        self.log('val_epoch_acc', epoch_acc)

    def _log_points(self, pos, cor_pred, batch_idx):
        tensorboard = self.logger.experiment
        pos = torch.unsqueeze(pos, 0)

        color = torch.empty_like(pos)
        cor_pred = cor_pred.long()
        for i in range(color.shape[1]):
            color[0, i, :] = self.colors[cor_pred[i]]

        config_pc = {'material': {'cls': 'PointsMaterial', 'size': 5}}

        tensorboard.add_mesh(f'val_mesh_{batch_idx}', pos,
                             colors=color,
                             config_dict=config_pc, global_step=self.global_step)


class EdgesClassification(NodesClassification, pl.LightningModule):
    def __init__(self, model, optimizer=None, logger=None):
        super(EdgesClassification, self).__init__(model=model,
                                                  optimizer=optimizer,
                                                  logger=logger)

    def training_step(self, batch, batch_idx):
        # to generalize
        batch = self.net(batch)
        logits = batch.out[:, 0]
        logits = torch.sigmoid(logits)
        pred = logits > 0.5
        loss = F.binary_cross_entropy(logits, batch.edge_y.float())

        glob_acc = self.micro_accuracy(pred, batch.edge_y)
        class_acc = self.macro_accuracy(pred, batch.edge_y)

        self.log('train_loss', loss)
        self.log('train_global_acc', glob_acc)
        self.log('train_class_acc', class_acc)
        return loss

    def _val_forward(self, val_batch):
        # to generalize
        val_batch = self.net(val_batch)
        logits = val_batch.out[:, 0]
        logits = torch.sigmoid(logits)
        pred = logits > 0.5
        loss = F.binary_cross_entropy(logits, val_batch.edge_y.float())
        return pred, loss
