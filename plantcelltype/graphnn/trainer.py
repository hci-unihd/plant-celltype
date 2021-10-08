import copy
import itertools
import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers

from plantcelltype.graphnn.pl_models import NodesClassification, EdgesClassification
from pctg_benchmark.loaders.torch_loader import PCTGSimpleSplit, PCTGCrossValidationSplit
from plantcelltype.utils.utils import print_config


loaders = {'PCTGSimpleSplit': PCTGSimpleSplit,
           'PCTGCrossValidationSplit': PCTGCrossValidationSplit}


def add_home_path(path):
    home_dir = os.path.expanduser('~')
    return f"{home_dir}{path}"


def get_model(config, in_features, in_edges_attr):
    if config['mode'] == 'NodesClassification':
        return NodesClassification(**config['module'])
    elif config['mode'] == 'EdgesClassification':
        return EdgesClassification(**config['module'])
    else:
        raise NotImplementedError


def get_loaders(config):
    name = config['name']
    test_loader = loaders[name](**config['test'])
    train_loader = loaders['name'](**config['train'])
    in_edges_attr = train_loader.in_edges_attr
    in_features = train_loader.in_features
    return test_loader, train_loader, in_features, in_edges_attr


def get_logger(config):
    config['trainer']['logger'] = pl_loggers.TensorBoardLogger(**config['logs'])
    return config


def simple_train(config):
    pl.seed_everything(42, workers=True)
    test_loader, train_loader, in_features, in_edges_attr = get_loaders(config)

    model = get_model(config, in_features, in_edges_attr)
    config = get_logger(config)
    trainer = pl.Trainer(**config['trainer'])
    trainer.fit(model, train_loader, test_loader)

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

    return checkpoint_path


def setup_cross_validation(config):
    data_location = config['loader']['path']
    list_path = f'{data_location}/list_data.csv'
    split = config['cross_validation'].get('split', 5)
    seed = config['cross_validation'].get('seed', 0)

    config['loader']['mode'] = 'split'
    return config, splits


def cross_validation_train(config):
    config, splits = setup_cross_validation(config)
    run_name = config['logs']['name']
    for key, split in splits.items():
        config['logs']['name'] = f'{run_name}_split{key}'
        config['loader']['path'] = split
        config['split_id'] = key
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
    if 'cross_validation' in config:
        cross_validation_train(config)
    else:
        simple_train(config)
