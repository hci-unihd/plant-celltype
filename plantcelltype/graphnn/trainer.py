import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
from plantcelltype.graphnn.data_loader import build_geometric_loaders
from plantcelltype.graphnn.pl_models import NodesClassification, EdgesClassification


def add_home_path(path):
    home_dir = os.path.expanduser('~')
    return f"{home_dir}{path}"


def get_model(config):
    if config['mode'] == 'NodesClassification':
        return NodesClassification(**config['module'])
    elif config['mode'] == 'EdgesClassification':
        return EdgesClassification(**config['module'])
    else:
        raise NotImplementedError


def get_loaders(config):
    (test_loader,
     train_loader,
     in_features,
     in_edges) = build_geometric_loaders(**config['loader'])

    config['module']['model']['kwargs']['in_features'] = in_features
    if in_edges is not None:
        config['module']['model']['kwargs']['in_edges'] = in_edges
    return test_loader, train_loader, config


def get_logger(config):
    config['trainer']['logger'] = pl_loggers.TensorBoardLogger(**config['logs'])
    return config


def simple_train(config):
    test_loader, train_loader, config = get_loaders(config)
    model = get_model(config)
    config = get_logger(config)

    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(model, train_loader, test_loader)


def cross_validation_train(config, num_splits):
    pass
