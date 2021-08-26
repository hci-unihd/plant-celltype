from plantcelltype.graphnn.trainer import simple_train
from plantcelltype.utils.io import load_yaml
import argparse


def parser():
    _parser = argparse.ArgumentParser(description='plant-celltype training config')
    _parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = _parser.parse_args()
    return args


_args = parser()
config = load_yaml(_args.config)
print('config: ', config)
simple_train(config)
