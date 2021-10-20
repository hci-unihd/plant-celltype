from pctg_benchmark.utils.utils import get_basic_loader_config
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train
import copy


template_config_paths = ('./features_importance/deeper_gcn.yaml',
                         './features_importance/gcn.yaml',
                         )

template_perturbations = ({'name': 'SetToValue'},
                          {'name': 'SetToRandom'},
                          {'name': 'RandomShuffle'})

features_groups = (('baseline', ),  # nothing will be changed compared to the default config
                   ('com_grs', ),
                   ('lrs_orientation_axis1_grs',
                    'lrs_orientation_axis2_grs',
                    'lrs_orientation_axis3_grs'),
                   ('lrs_axis1_grs',
                    'lrs_axis2_grs',
                    'lrs_axis3_grs'),
                   ('pca_orientation_axis1_grs',
                    'pca_orientation_axis3_grs',
                    'pca_orientation_axis3_grs'),
                   ('degree_centrality', ),
                   ('rw_centrality', ),
                   ('hops_to_bg', ),
                   ('volume_um',
                    'surface_um'),
                   ('length_axis1_grs',
                    'length_axis2_grs',
                    'length_axis3_grs'),
                   ('lrs_axis12_dot_grs',
                    'lrs_axis2_angle_grs'),
                   ('lrs_proj_axis1_grs',
                    'lrs_proj_axis2_grs',
                    'lrs_proj_axis3_grs'),
                   ('pca_proj_axis1_grs',
                    'pca_proj_axis2_grs',
                    'pca_proj_axis3_grs'),
                   ('pca_explained_variance_grs', ),
                   ('proj_length_unit_sphere', )
                   )

for template_config_path in template_config_paths:
    template_config = load_yaml(template_config_path)
    dataset_config = get_basic_loader_config('dataset')
    for template_perturbation in template_perturbations:
        for group in features_groups:
            template_config_copy = copy.deepcopy(template_config)
            template_config_copy['logs']['name'] += f"_{template_perturbation['name']}_{'_'.join(group)}"
            # experiment setup
            dataset_config_copy = copy.deepcopy(dataset_config)
            for i, feat_config in enumerate(dataset_config_copy['node_features']):
                for feat in group:
                    if feat_config['name'] == feat:
                        feat_config['pre_transform'].insert(0, template_perturbation)
                        dataset_config_copy['node_features'][i] = feat_config

            for dataset in ['train_dataset', 'test_dataset', 'val_dataset']:
                template_config_copy['loader'][dataset]['raw_transform_config'] = dataset_config_copy
            train(config=template_config_copy)
