from pctg_benchmark.utils.utils import get_basic_loader_config
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train
import copy

template_config_paths = ('./features_importance/deeper_gcn.yaml',
                         './features_importance/gcn.yaml',
                         )

feat_mappings = {'lrs_orientation_axis1_grs': {'name': 'lrs_axis1_grs',
                                               'pre_transform': [{'name': 'ToUnitVector'},
                                                                 {'name': 'ToTorchTensor'}]},
                 'lrs_orientation_axis2_grs': {'name': 'lrs_axis2_grs',
                                               'pre_transform': [{'name': 'ToUnitVector'},
                                                                 {'name': 'ToTorchTensor'}]},
                 'lrs_orientation_axis3_grs': {'name': 'lrs_axis3_grs',
                                               'pre_transform': [{'name': 'ToUnitVector'},
                                                                 {'name': 'ToTorchTensor'}]},
                 'pca_orientation_axis1_grs': {'name': 'pca_axis1_grs',
                                               'pre_transform': [{'name': 'ToUnitVector'},
                                                                 {'name': 'ToTorchTensor'}]},
                 'pca_orientation_axis2_grs': {'name': 'pca_axis2_grs',
                                               'pre_transform': [{'name': 'ToUnitVector'},
                                                                 {'name': 'ToTorchTensor'}]},
                 'pca_orientation_axis3_grs': {'name': 'pca_axis3_grs',
                                               'pre_transform': [{'name': 'ToUnitVector'},
                                                                 {'name': 'ToTorchTensor'}]},
                 }

for template_config_path in template_config_paths:
    template_config = load_yaml(template_config_path)
    train(config=template_config)  # standard training

    dataset_config = get_basic_loader_config('dataset')

    for i, feat in enumerate(dataset_config['node_features']):
        if feat['name'] in feat_mappings.keys():
            dataset_config['node_features'][i].update(feat_mappings[feat['name']])

    for dataset in ['train_dataset', 'test_dataset', 'val_dataset']:
        dataset_config['loader'][dataset]['raw_transform_config'] = dataset_config

    template_config['logs']['name'] += f"_axis"
    train(config=template_config)  # standard training
