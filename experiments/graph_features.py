from ctg_benchmark.utils.utils import get_basic_loader_config
from ctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train

template_config_paths = ('./graph_features/gcn.yaml',
                         './graph_features/deeper_gcn.yaml',
                         )

feat_mappings = [({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ClipQuantile'},
                                                           {'name': 'Zscore'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ClipQuantile'},
                                                       {'name': 'Zscore'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 5, 'extreme': [0, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops5_01'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'Zscore'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'Zscore'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 5, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops5_NoClip'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ClipQuantile'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ClipQuantile'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 5, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops5_No_Zscore'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 5, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops5_No_Zscore_NoClip'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ClipQuantile'},
                                                           {'name': 'Zscore'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ClipQuantile'},
                                                       {'name': 'Zscore'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 5, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops5'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ClipQuantile'},
                                                           {'name': 'Zscore'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ClipQuantile'},
                                                       {'name': 'Zscore'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 4, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops4'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ClipQuantile'},
                                                           {'name': 'Zscore'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ClipQuantile'},
                                                       {'name': 'Zscore'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 3, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops3'),
                 ({'degree_centrality': {'name': 'degree_centrality',
                                         'pre_transform': [{'name': 'ClipQuantile'},
                                                           {'name': 'Zscore'},
                                                           {'name': 'ToTorchTensor'}]},
                   'rw_centrality': {'name': 'rw_centrality',
                                     'pre_transform': [{'name': 'ClipQuantile'},
                                                       {'name': 'Zscore'},
                                                       {'name': 'ToTorchTensor'}]},
                   'hops_to_bg': {'name': 'hops_to_bg',
                                  'pre_transform': [{'name': 'ToOnehot', 'max_channel': 2, 'extreme': [-1, 1]},
                                                    {'name': 'ToTorchTensor'}]}
                   }
                  , '_Hops2'),
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
