from plantcelltype.pipeline import run_data_processing, run_train_data_processing
from ctg_benchmark.utils.io import load_yaml

from plantcelltype.utils.utils import parser


def main():
    _args = parser()
    _config = load_yaml(_args.config)
    mode = _config.get('mode', 'main')
    if mode == 'main':
        run_data_processing(_config)

    elif mode == 'train':
        run_train_data_processing(_config)


if __name__ == '__main__':
    main()
    