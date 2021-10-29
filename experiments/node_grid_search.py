from plantcelltype.graphnn.trainer import grid_search_train
from pctg_benchmark.utils.io import load_yaml

template_config_paths = ('./experiments/node_grid_search/predict_from_segmentation.yaml',
                         './experiments/node_grid_search/gat.yaml',
                         './experiments/node_grid_search/gatv2.yaml',
                         './experiments/node_grid_search/gcn.yaml',
                         './experiments/node_grid_search/gcnii.yaml',
                         './experiments/node_grid_search/tgcn.yaml',
                         './experiments/node_grid_search/tgcn_edge.yaml',
                         )


for path in template_config_paths:
    config = load_yaml(path)
    config_grid_search = config.pop('grid_search')
    grid_search_train(config=config, kwargs=config_grid_search)
