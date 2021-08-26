from plantcelltype.graphnn.trainer import train
from plantcelltype.utils.io import load_yaml
import argparse


def print_config(config, indentation=0):
    spaces = indentation * '  '
    for key, value in config.items():
        if isinstance(value, dict):
            print(f'{spaces}{key}:')
            print_config(value, indentation + 1)
        else:
            print(f'{spaces}{key}: {value}')


def parser():
    _parser = argparse.ArgumentParser(description='plant-celltype training config')
    _parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    args = _parser.parse_args()
    return args


_args = parser()
_config = load_yaml(_args.config)
print_config(_config)
train(_config)
