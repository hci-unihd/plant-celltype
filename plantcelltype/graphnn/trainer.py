from plantcelltype.graphnn.pl_models import GraphClassification
import pytorch_lightning as pl
from plantcelltype.graphnn.data_loader import get_random_split

files_path = "/home/lcerrone/PycharmProjects/plant-celltype/data/ovules-celltype-processed/**/*.h5"
test_loader, train_loader = get_random_split(files_path)

model = GraphClassification(model_name='TGCN3',
                            model_kwargs={'in_features': 13, 'out_features': 10, 'hidden_feat': [256, 256]})
trainer = pl.Trainer()
trainer.fit(model, train_loader, test_loader)
