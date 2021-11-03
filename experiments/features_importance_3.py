from pctg_benchmark.utils.utils import get_basic_loader_config
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train
import copy
import os

template_config_paths = ('./features_importance/gcn.yaml',
                         './features_importance/deeper_gcn.yaml',
                         )

template_perturbations = ({'name': 'SetToValue'},
                          {'name': 'SetToRandom'},
                          {'name': 'RandomShuffle'})

invariant = ['com_voxels',
             'degree_centrality',
             'hops_to_bg',
             'lrs_axis2_angle_grs',
             'rw_centrality',
             'surface_um',
             'surface_voxels',
             'volume_um',
             'volume_voxels']

approx_invariant = ['length_axis1_grs',
                    'length_axis2_grs',
                    'length_axis3_grs',
                    'pca_explained_variance_grs',
                    'pca_orientation_axis1_grs',
                    'pca_orientation_axis2_grs',
                    'pca_orientation_axis3_grs',
                    'pca_proj_axis1_grs',
                    'pca_proj_axis2_grs',
                    'pca_proj_axis3_grs']

not_invariant = ['com_grs',
                 'com_proj_grs',
                 'lrs_axis12_dot_grs',
                 'lrs_orientation_axis1_grs',
                 'lrs_orientation_axis2_grs',
                 'lrs_orientation_axis3_grs',
                 'lrs_proj_axis1_grs',
                 'lrs_proj_axis2_grs',
                 'lrs_proj_axis3_grs']

features_groups = (('inv', invariant),
                   ('not_inv', not_invariant),
                   ('approx', approx_invariant),
                   ('inv_not_inv', invariant + not_invariant),
                   ('inv_approx', invariant + approx_invariant),
                   ('not_inv_approx', not_invariant + approx_invariant),
                   )


for template_config_path in template_config_paths:
    template_config = load_yaml(template_config_path)

    template_config['logs']['save_dir'] = f"{os.path.expanduser('~')}/results/plant-ct-logs/ablation/features_importance_group/"
    dataset_config = get_basic_loader_config('dataset')
    for template_perturbation in template_perturbations:
        for name, group in features_groups:
            template_config_copy = copy.deepcopy(template_config)
            template_config_copy['logs']['name'] += f'_{name}'
            # experiment setup
            dataset_config_copy = copy.deepcopy(dataset_config)
            for i, feat_config in enumerate(dataset_config_copy['node_features']):
                for feat in group:
                    if feat_config['name'] == feat:
                        feat_config['pre_transform'].insert(0, template_perturbation)
                        dataset_config_copy['node_features'][i] = feat_config

            for dataset in ['train_dataset', 'test_dataset', 'val_dataset']:
                if dataset in template_config_copy['loader']:
                    template_config_copy['loader'][dataset]['raw_transform_config'] = dataset_config_copy
            train(config=template_config_copy)
