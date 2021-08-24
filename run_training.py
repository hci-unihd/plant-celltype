from plantcelltype.graphnn.trainer import main_train

config = {'files_path': "/data/ovules/ovules-celltype-processed/**/*.h5",
          'logs_path': "/results/plant-ct-logs/",
          'run_keyword': "",
          'load_edge_attr': True,
          'lr': 1e-4,
          'wd': 1e-6,
          'model_name': 'TGCN2',
          'model_kwargs': {'in_features': None,
                           'out_features': 10,
                           'hidden_feat': 100,
                           'in_edges': 100}}

main_train(config=config)
