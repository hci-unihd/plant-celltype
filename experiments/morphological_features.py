from pctg_benchmark.utils.utils import get_basic_loader_config
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train

template_config_paths = ('./morphological_features/gcn.yaml',
                         './morphological_features/deeper_gcn.yaml',
                         )

feat_mappings = [({'volume_um': {'name': 'volume_um',
                                 'pre_transform': [{'name': 'ClipQuantile'},
                                                   {'name': 'Zscore'},
                                                   {'name': 'ToTorchTensor'}]},
                   'surface_um': {'name': 'surface_um',
                                  'pre_transform': [{'name': 'ClipQuantile'},
                                                    {'name': 'Zscore'},
                                                    {'name': 'ToTorchTensor'}]},
                   'length_axis1_grs': {'name': 'length_axis1_grs',
                                        'pre_transform': [{'name': 'ClipQuantile'},
                                                          {'name': 'Zscore'},
                                                          {'name': 'ToTorchTensor'}]},
                   'length_axis2_grs': {'name': 'length_axis2_grs',
                                        'pre_transform': [{'name': 'ClipQuantile'},
                                                          {'name': 'Zscore'},
                                                          {'name': 'ToTorchTensor'}]},
                   'length_axis3_grs': {'name': 'length_axis3_grs',
                                        'pre_transform': [{'name': 'ClipQuantile'},
                                                          {'name': 'Zscore'},
                                                          {'name': 'ToTorchTensor'}]}
                   }, '_Base'),
                 ({'volume_um': {'name': 'volume_um',
                                 'pre_transform': [{'name': 'Zscore'},
                                                   {'name': 'ToTorchTensor'}]},
                   'surface_um': {'name': 'surface_um',
                                  'pre_transform': [{'name': 'Zscore'},
                                                    {'name': 'ToTorchTensor'}]},
                   'length_axis1_grs': {'name': 'length_axis1_grs',
                                        'pre_transform': [{'name': 'Zscore'},
                                                          {'name': 'ToTorchTensor'}]},
                   'length_axis2_grs': {'name': 'length_axis2_grs',
                                        'pre_transform': [{'name': 'Zscore'},
                                                          {'name': 'ToTorchTensor'}]},
                   'length_axis3_grs': {'name': 'length_axis3_grs',
                                        'pre_transform': [{'name': 'Zscore'},
                                                          {'name': 'ToTorchTensor'}]}
                   }, '_No_Clip'),
                 ({'volume_um': {'name': 'volume_um',
                                 'pre_transform': [{'name': 'ClipQuantile'},
                                                   {'name': 'ToTorchTensor'}]},
                   'surface_um': {'name': 'surface_um',
                                  'pre_transform': [{'name': 'ClipQuantile'},
                                                    {'name': 'ToTorchTensor'}]},
                   'length_axis1_grs': {'name': 'length_axis1_grs',
                                        'pre_transform': [{'name': 'ClipQuantile'},
                                                          {'name': 'ToTorchTensor'}]},
                   'length_axis2_grs': {'name': 'length_axis2_grs',
                                        'pre_transform': [{'name': 'ClipQuantile'},
                                                          {'name': 'ToTorchTensor'}]},
                   'length_axis3_grs': {'name': 'length_axis3_grs',
                                        'pre_transform': [{'name': 'ClipQuantile'},
                                                          {'name': 'ToTorchTensor'}]}
                   }, '_No_Zscore'),
                 ({'volume_um': {'name': 'volume_um',
                                 'pre_transform': [{'name': 'ToTorchTensor'}]},
                   'surface_um': {'name': 'surface_um',
                                  'pre_transform': [{'name': 'ToTorchTensor'}]},
                   'length_axis1_grs': {'name': 'length_axis1_grs',
                                        'pre_transform': [{'name': 'ToTorchTensor'}]},
                   'length_axis2_grs': {'name': 'length_axis2_grs',
                                        'pre_transform': [{'name': 'ToTorchTensor'}]},
                   'length_axis3_grs': {'name': 'length_axis3_grs',
                                        'pre_transform': [{'name': 'ToTorchTensor'}]}
                   }, '_No_Zscore_No_Clip')

                 ]

for template_config_path in template_config_paths:
    for feat_mapping, name in feat_mappings:
        template_config = load_yaml(template_config_path)
        dataset_config = get_basic_loader_config('dataset')
        for i, feat in enumerate(dataset_config['node_features']):
            if feat['name'] in feat_mapping.keys():
                dataset_config['node_features'][i].update(feat_mapping[feat['name']])

        for dataset in ['train_dataset', 'test_dataset', 'val_dataset']:
            if dataset in template_config['loader']:
                template_config['loader'][dataset]['raw_transform_config'] = dataset_config

        template_config['logs']['name'] += name
        train(config=template_config)  # perturbed training
