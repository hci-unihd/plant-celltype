import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
from plantcelltype.graphnn.data_loader import build_geometric_loaders
from plantcelltype.graphnn.pl_models import NodesClassification


def main_train(config):
    home_dir = os.path.expanduser('~')
    files_path = os.path.join(home_dir, config['files_path'])
    model_kwargs = config['model_kwargs']

    test_loader, train_loader, n_feat = build_geometric_loaders(files_path)

    model_kwargs['in_features'] = n_feat
    model = NodesClassification(model_name=config['model_name'],
                                model_kwargs=model_kwargs)

    # trainer = pl.Trainer(gpus=1)
    run_name = f"{config['model_name']}_{config['run_keyword']}"
    tb_logger = pl_loggers.TensorBoardLogger(config['logs_path'], name=run_name)
    trainer = pl.Trainer(logger=tb_logger)
    trainer.fit(model, train_loader, test_loader)
