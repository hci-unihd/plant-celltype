import copy
import itertools
import os
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import yaml
from pytorch_lightning import loggers as pl_loggers
from torch_geometric.loader import DataLoader
from plantcelltype.graphnn.pl_models import NodesClassification, EdgesClassification
from pctg_benchmark.loaders.torch_loader import PCTGSimpleSplit, PCTGCrossValidationSplit
from plantcelltype.utils.utils import print_config


datasets = {'simple': PCTGSimpleSplit,
            'cross_validation': PCTGCrossValidationSplit}


class LogConfigCallback(Callback):
    def __init__(self, config):
        self.config = config

    def _save(self, trainer, model) -> None:
        config = copy.deepcopy(self.config)
        version = f'version_{trainer.logger.version}'
        checkpoint_path = os.path.join(trainer.logger.save_dir,
                                       trainer.logger.name,
                                       version)

        config['run'] = {'save_dir': trainer.logger.save_dir,
                         'name': trainer.logger.name,
                         'version': trainer.logger.version,
                         'results': model.saved_metrics}

        del config['trainer']['logger']
        with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as outfile:
            yaml.dump(config, outfile)

    def on_validation_end(self, trainer, model) -> None:
        self._save(trainer, model)

    def on_test_end(self, trainer, model) -> None:
        self._save(trainer, model)


def get_model(config, in_features, in_edges_attr):
    config['module']['model']['kwargs']['in_features'] = in_features
    if 'in_edges_attr' in config['module']['model']['kwargs']:
        config['module']['model']['kwargs']['in_edges_attr'] = in_edges_attr

    if config['mode'] == 'NodesClassification':
        return NodesClassification(**config['module'])
    elif config['mode'] == 'EdgesClassification':
        return EdgesClassification(**config['module'])
    else:
        raise NotImplementedError


def get_loaders(config):
    val_dataset = datasets[config['mode']](**config['val_dataset'])
    train_dataset = datasets[config['mode']](**config['train_dataset'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config['val_batch_size'],
                            num_workers=config['num_workers'],
                            shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=config['train_batch_size'],
                              num_workers=config['num_workers'],
                              shuffle=True)

    in_edges_attr = train_dataset.in_edges_attr
    in_features = train_dataset.in_features
    return val_loader, train_loader, in_features, in_edges_attr


def get_logger(config):
    config['trainer']['logger'] = pl_loggers.TensorBoardLogger(**config['logs'])
    return config


def run_test(trainer, model, config):
    test_dataset = datasets[config['mode']](**config['test_dataset'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config['val_batch_size'],
                             num_workers=config['num_workers'],
                             shuffle=False)
    trainer.test(model, test_loader)


def simple_train(config):
    pl.seed_everything(config.get('seed', 42), workers=True)
    val_loader, train_loader, in_features, in_edges_attr = get_loaders(config['loader'])

    model = get_model(config, in_features, in_edges_attr)
    config = get_logger(config)
    trainer = pl.Trainer(**config['trainer'],
                         callbacks=[LogConfigCallback(config)])

    trainer.fit(model, train_loader, val_loader)
    run_test(trainer, model, config['loader'])

    version = f'version_{trainer.logger.version}'
    checkpoint_path = os.path.join(trainer.logger.save_dir,
                                   trainer.logger.name,
                                   version)
    return checkpoint_path


def cross_validation_train(config):
    n_splits = config['loader']['n_splits']
    run_name = config['logs']['name']
    for split in range(n_splits):
        config['logs']['name'] = f'{run_name}_split{split}'
        config['split_id'] = split
        simple_train(config)


def update_nested_dict(base, key, value):
    keys = key.split('/')
    key0, _key = keys[0], '/'.join(keys[1:])
    if len(keys) == 1:
        base.update({key0: value})
    else:
        if key0 not in base:
            base.update({key0: {}})
        up_config = update_nested_dict(base[key0], '/'.join(keys[1:]), value)
        base.update({key0: up_config})
    return base


def grid_search_train(config, kwargs):
    all_config = []
    for new_params in itertools.product(*kwargs.values()):
        _config = copy.deepcopy(config)
        new_name = _config['logs']['name']

        for value, key in zip(new_params, kwargs.keys()):
            _config = update_nested_dict(_config, key, value)
            key_final = key.split('/')[-1]
            new_name += f'_{key_final}:{value}'
        _config['logs']['name'] = new_name
        all_config.append(_config)

    for _config in all_config:
        train(_config)


def train(config):
    print_config(config)
    if config['loader']['mode'] == 'cross_validation':
        cross_validation_train(config)
    elif config['loader']['mode'] == 'simple':
        simple_train(config)
    else:
        raise NotImplementedError
