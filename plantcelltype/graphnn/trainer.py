import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
from plantcelltype.graphnn.data_loader import build_geometric_loaders
from plantcelltype.graphnn.pl_models import NodesClassification, EdgesClassification


def main_train(config):
    home_dir = os.path.expanduser('~')
    config['loader']['path'] = f"{home_dir}{config['loader']['path']}"
    logs_path = f"{home_dir}{config['logs']['path']}"
    model_kwargs = config['pl_lightning']['config_model']['kwargs']

    (test_loader,
     train_loader,
     n_feat,
     n_edge_feat) = build_geometric_loaders(**config['loader'])

    model_kwargs['in_features'] = n_feat
    if config['loader']['load_edge_attr']:
        model_kwargs['in_edges'] = n_edge_feat

    config['pl_lightning']['config_model']['kwargs'] = model_kwargs
    if config['mode'] == 'NodesClassification':
        model = NodesClassification(**config['pl_lightning'])

    elif config['mode'] == 'EdgesClassification':
        model = EdgesClassification(**config['pl_lightning'])

    else:
        raise NotImplementedError

    run_name = f"{config['pl_lightning']['config_model']['name']}_{config['logs']['run_keyword']}"
    tb_logger = pl_loggers.TensorBoardLogger(logs_path, name=run_name)

    config['trainer']['logger'] = tb_logger
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(model, train_loader, test_loader)
