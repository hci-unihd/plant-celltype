from ctg_benchmark.utils.utils import get_basic_loader_config
from ctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train
import os


def train_feat_perturbations(features_groups, template_perturbations, template_config_paths, save_dir):
    for template_perturbation in template_perturbations:
        for template_config_path in template_config_paths:
            for name, group in features_groups:

                template_config = load_yaml(template_config_path)
                template_config['logs']['save_dir'] = f"{os.path.expanduser('~')}{save_dir}"
                dataset_config = get_basic_loader_config('dataset')

                template_config['logs']['name'] += f"_{template_perturbation['name']}_{name}"
                # experiment setup
                for i, feat_config in enumerate(dataset_config['node_features']):
                    for feat in group:
                        if feat_config['name'] == feat:
                            feat_config['pre_transform'].insert(0, template_perturbation)
                            dataset_config['node_features'][i] = feat_config

                for dataset in ['train_dataset', 'test_dataset', 'val_dataset']:
                    if dataset in template_config['loader']:
                        template_config['loader'][dataset]['raw_transform_config'] = dataset_config
                train(config=template_config)


if __name__ == '__main__':
    _template_config_paths = ('./features_importance/gcn.yaml',
                              './features_importance/deeper_gcn.yaml',
                              )

    _template_perturbations = ({'name': 'SetToValue'},
                               {'name': 'SetToRandom'},
                               {'name': 'RandomShuffle'}
                               )

    pos = ['com_grs', ]

    graph = ['degree_centrality',
             'hops_to_bg',
             'rw_centrality', ]

    morphology = ['surface_um',
                  'volume_um', ]

    lrs = ['lrs_axis12_dot_grs',
           'lrs_axis2_angle_grs',
           'lrs_orientation_axis1_grs',
           'lrs_orientation_axis2_grs',
           'lrs_orientation_axis3_grs',
           'lrs_proj_axis1_grs',
           'lrs_proj_axis2_grs',
           'lrs_proj_axis3_grs',
           'length_axis1_grs',
           'length_axis2_grs',
           'length_axis3_grs', ]

    pca = ['pca_explained_variance_grs',
           'pca_orientation_axis1_grs',
           'pca_orientation_axis2_grs',
           'pca_orientation_axis3_grs',
           'pca_proj_axis1_grs',
           'pca_proj_axis2_grs',
           'pca_proj_axis3_grs', ]

    _features_groups = (('baseline', ()),
                        ('pos', pos),
                        ('graph', graph),
                        ('morphology', morphology),
                        ('lrs', lrs),
                        ('pca', pca)
                        )

    _save_dir = '/results/plant-ct-logs/ablation/features_importance_type/'

    train_feat_perturbations(_features_groups, _template_perturbations, _template_config_paths, _save_dir)
