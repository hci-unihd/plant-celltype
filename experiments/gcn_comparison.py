from pctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train

template_config_paths = ('./gcn_comparison/tg_gcn.yaml',
                         './gcn_comparison/gcn.yaml',
                         )

for template_config_path in template_config_paths:
    template_config = load_yaml(template_config_path)
    train(config=template_config)
