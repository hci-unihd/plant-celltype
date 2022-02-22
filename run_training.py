from plantcelltype.graphnn.trainer import train, grid_search_train
from ctg_benchmark.utils.io import load_yaml

from plantcelltype.utils.utils import parser

if __name__ == '__main__':
    _args = parser()
    _config = load_yaml(_args.config)

    if 'grid_search' not in _config:
        train(_config)
    else:
        config_grid_search = _config.pop('grid_search')
        grid_search_train(config=_config, kwargs=config_grid_search)
