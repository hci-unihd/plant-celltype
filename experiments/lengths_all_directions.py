from pctg_benchmark.utils.utils import get_basic_loader_config
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train

template_config_paths = ('./lengths_all_directions/gcn.yaml',
                         './lengths_all_directions/deeper_gcn.yaml',
                         )

perturbation = {'name': 'proj_length_unit_sphere',
                'pre_transform': [{'name': 'ClipQuantile'},
                                  {'name': 'Zscore'},
                                  {'name': 'ToTorchTensor'}]}
name = '_All_Lengths'
for template_config_path in template_config_paths:
    template_config = load_yaml(template_config_path)
    train(config=template_config)  # perturbed training

    dataset_config = get_basic_loader_config('dataset')
    dataset_config['node_features'].append(perturbation)
    for dataset in ['train_dataset', 'test_dataset', 'val_dataset']:
        if dataset in template_config['loader']:
            template_config['loader'][dataset]['raw_transform_config'] = dataset_config

    template_config['logs']['name'] += name
    train(config=template_config)  # perturbed training
