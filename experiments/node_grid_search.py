from plantcelltype.graphnn.trainer import grid_search_train
from pctg_benchmark.utils.io import load_yaml

template_config_paths = ('./node_grid_search/deeper_gcn.yaml',
                         './node_grid_search/deeper_gcn_no_edges.yaml'
                         './node_grid_search/gat.yaml',
                         './node_grid_search/gatv2.yaml',
                         './node_grid_search/gcn.yaml',
                         './node_grid_search/gcnii.yaml',
                         './node_grid_search/gin.yaml',
                         './node_grid_search/graphsage.yaml',
                         './node_grid_search/tgcn.yaml',
                         './node_grid_search/tgcn_no_edge.yaml',
                         )


for path in template_config_paths:
    config = load_yaml(path)
    config_grid_search = config.pop('grid_search')
    grid_search_train(config=config, kwargs=config_grid_search)
