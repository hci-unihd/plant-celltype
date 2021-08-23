from plantcelltype.graphnn.trainer import main_train

config = {'files_path': "/data/ovules/ovules-celltype-processed/**/*.h5",
          'logs_path': "/data/results/plant-ct-logs/",
          'run_keyword': "",
          'model_name': 'TGCN2',
          'model_kwargs': {'in_features': None, 'out_features': 10, 'hidden_feat': 256}}

main_train(config=config)
