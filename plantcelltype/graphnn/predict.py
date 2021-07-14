from plantcelltype.graphnn.pl_models import NodesClassification, EdgesClassification
import pytorch_lightning as pl
from plantcelltype.graphnn.data_loader import get_random_split
from plantcelltype.utils import create_h5

files_path = "/home/lcerrone/PycharmProjects/plant-celltype/data/ovules-celltype-processed/**/*.h5"
check_point_path = "/home/lcerrone/PycharmProjects/plant-celltype/plantcelltype/graphnn/lightning_logs/version_56/checkpoints/epoch=8-step=512.ckpt"
test_loader, _ = get_random_split(files_path)
"""
model = GraphClassification(model_name='GCN3',
                            model_kwargs={'in_features': 13, 'out_features': 10, 'hidden_feat': [256, 256]})
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, test_loader)
"""
#model = EdgesClassification(model_name='ETGCN2',
#                            model_kwargs={'in_features': 13, 'out_features': 2, 'hidden_feat': 256})
model = EdgesClassification.load_from_checkpoint(check_point_path,
                                                 model_name='ETGCN2',
                                                 model_kwargs={'in_features': 13, 'out_features': 1, 'hidden_feat': 256})
for data in test_loader:
    pred, _ = model.forward(data)
    logit = pred.out.cpu().data.numpy()
    print(data.file_path)
    create_h5(data.file_path[0], logit, key='edges_predictions', voxel_size=None)
