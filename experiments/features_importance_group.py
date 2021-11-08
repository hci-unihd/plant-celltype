from features_importance_type import train_feat_perturbations

if __name__ == '__main__':
    _template_config_paths = ('./features_importance/gcn.yaml',
                              './features_importance/deeper_gcn.yaml',
                              )

    _template_perturbations = ({'name': 'SetToValue'},
                               {'name': 'SetToRandom'},
                               {'name': 'RandomShuffle'}
                               )

    invariant = ['degree_centrality',
                 'hops_to_bg',
                 'lrs_axis2_angle_grs',
                 'rw_centrality',
                 'surface_um',
                 'volume_um']

    approx_invariant = ['length_axis1_grs',
                        'length_axis2_grs',
                        'length_axis3_grs', ]

    not_invariant = ['com_grs',
                     'lrs_axis12_dot_grs',
                     'lrs_orientation_axis1_grs',
                     'lrs_orientation_axis2_grs',
                     'lrs_orientation_axis3_grs',
                     'lrs_proj_axis1_grs',
                     'lrs_proj_axis2_grs',
                     'lrs_proj_axis3_grs',
                     'pca_explained_variance_grs',
                     'pca_orientation_axis1_grs',
                     'pca_orientation_axis2_grs',
                     'pca_orientation_axis3_grs',
                     'pca_proj_axis1_grs',
                     'pca_proj_axis2_grs',
                     'pca_proj_axis3_grs'
                     ]

    _features_groups = (('baseline', ()),
                        ('inv', invariant),
                        ('not_inv', not_invariant),
                        ('approx', approx_invariant),
                        ('inv_not_inv', invariant + not_invariant),
                        ('inv_approx', invariant + approx_invariant),
                        ('not_inv_approx', not_invariant + approx_invariant),
                        )

    _save_dir = '/results/plant-ct-logs/ablation/features_importance_group/'

    train_feat_perturbations(_features_groups, _template_perturbations, _template_config_paths, _save_dir)
