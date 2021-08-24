from plantcelltype.graphnn.trainer import main_train
import yaml
import argparse


def parser():
    _parser = argparse.ArgumentParser(description='plant-celltype training config')
    _parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = _parser.parse_args()
    return args


def load_config(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    return config


_args = parser()
_config = load_config(_args)

main_train(config=_config)
