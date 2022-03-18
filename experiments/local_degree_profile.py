from ctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train
from torch_geometric.transforms import LocalDegreeProfile, Compose
template_config_paths = ('./local_degree_profile/gcn.yaml',
                         './local_degree_profile/deeper_gcn.yaml',
                         )

name = '_local_degree_profile'
for template_config_path in template_config_paths:
    template_config = load_yaml(template_config_path)
    template_config['logs']['name'] += name
    template_config['loader']['train_dataset']['pre_transform'] = Compose([LocalDegreeProfile()])
    template_config['loader']['val_dataset']['pre_transform'] = Compose([LocalDegreeProfile()])
    train(config=template_config)  # perturbed training

    template_config = load_yaml(template_config_path)
    train(config=template_config)  # perturbed training
