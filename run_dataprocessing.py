from plantcelltype.run_pipeline import main, process_train_data
from ctg_benchmark.utils.io import load_yaml

from plantcelltype.utils.utils import parser

if __name__ == '__main__':
    _args = parser()
    _config = load_yaml(_args.config)
    mode = _config.get('mode', 'main')
    if mode == 'main':
        main(_config)
    elif mode == 'train':
        process_train_data(_config)
