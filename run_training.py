from plantcelltype.graphnn.trainer import main_train

config = {'files_path': "/data/ovules/ovules-celltype-processed/**/*.h5",
          'logs_path': "/results/plant-ct-logs/baseline/",
          'run_keyword': "default",
          'load_edge_attr': False,
          'lr': 1e-4,
          'wd': 1e-6,
          'model_name': 'TransformerGCN2',
          'model_kwargs': {'out_features': 10,
                           'hidden_feat': 128}}

main_train(config=config)
