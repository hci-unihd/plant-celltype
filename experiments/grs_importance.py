from ctg_benchmark.utils.io import load_yaml
from plantcelltype.graphnn.trainer import train

template_config_paths = ('./grs_importance/gcn.yaml',
                         './grs_importance/deeper_gcn.yaml',
                         )

grs_groups = ({'train': ('trivial_grs',), 'test': ('trivial_grs',)},
              {'train': ('label_grs_surface',), 'test': ('label_grs_surface',)},
              {'train': ('label_grs_funiculus',), 'test': ('label_grs_funiculus',)},
              {'train': ('es_trivial_grs',), 'test': ('es_trivial_grs',)},
              {'train': ('es_pca_grs',), 'test': ('es_pca_grs',)},
              {'train': ('label_grs_funiculus',
                         'es_trivial_grs',
                         'es_pca_grs'),
               'test': ('label_grs_surface',)},
              {'train': ('es_pca_grs',),
               'test': ('label_grs_surface',)},
              )

for template_config_path in template_config_paths:
    for grs_group in grs_groups:
        template_config = load_yaml(template_config_path)
        train_name, test_name = '_'.join(grs_group['train']), '_'.join(grs_group['test'])
        template_config['logs']['name'] += f'_train:{train_name}_test:{test_name}'
        template_config['loader']['train_dataset']['grs'] = grs_group['train']
        template_config['loader']['val_dataset']['grs'] = grs_group['test']
        train(config=template_config)
