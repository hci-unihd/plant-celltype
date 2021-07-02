from plantcelltype.graphnn.pl_models import GraphClassification, EdgesClassification
import pytorch_lightning as pl
from plantcelltype.graphnn.data_loader import get_random_split

files_path = "/home/lcerrone/PycharmProjects/plant-celltype/data/ovules-celltype-processed/**/*.h5"
test_loader, train_loader = get_random_split(files_path)
"""
model = GraphClassification(model_name='GCN3',
                            model_kwargs={'in_features': 13, 'out_features': 10, 'hidden_feat': [256, 256]})
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, test_loader)
"""
model = EdgesClassification(model_name='ETGCN2',
                            model_kwargs={'in_features': 13, 'out_features': 1, 'hidden_feat': 256})
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, test_loader)
