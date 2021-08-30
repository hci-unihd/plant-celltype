import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
import yaml
from plantcelltype.graphnn.data_loader import build_geometric_loaders, get_n_splits
from plantcelltype.graphnn.pl_models import NodesClassification, EdgesClassification

pl.seed_everything(42, workers=True)


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


def cross_validation_train(config):
    data_location = config['loader']['path']
    list_path = f'{data_location}/list_data.csv'
    split = config['cross_validation'].get('split', 5)
    seed = config['cross_validation'].get('seed', 0)
    splits = get_n_splits(data_location, list_path, number_split=split, seed=seed)

    config['loader']['mode'] = 'split'
    run_name = config['logs']['name']
    run_check_points = []
    for key, split in splits.items():
        config['logs']['name'] = f'{run_name}_split{key}'
        config['loader']['path'] = split
        run_check_points.append(simple_train(config))


def train(config):
    if 'cross_validation' in config:
        cross_validation_train(config)
    else:
        simple_train(config)
