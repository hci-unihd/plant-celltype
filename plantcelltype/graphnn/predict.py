from plantcelltype.graphnn.data_loader import create_loaders
from plantcelltype.graphnn.trainer import get_model
from plantcelltype.utils.io import load_yaml
from plantcelltype.utils import create_h5
import glob
import torch


def build_test_loader(config, glob_paths=True):
    if glob_paths:
        config['files_list'] = glob.glob(config['files_list'])
    return create_loaders(**config)


config_path = "/home/lcerrone/PycharmProjects/plant-celltype/config/node_predictions/depp_gcn.yaml"
predict_config = load_yaml(config_path=config_path)
check_point = predict_config['checkpoint']
check_point_config = f'{check_point}/config.yaml'
check_point_weights = f'{check_point}/checkpoints/*ckpt'
check_point_weights = glob.glob(check_point_weights)[0]
model_config = load_yaml(check_point_config)


test_loader = build_test_loader(predict_config['loader'])
model = get_model(model_config)
model = model.load_from_checkpoint(check_point_weights)

for data in test_loader:
    data, _ = model.forward(data)
    logits = torch.log_softmax(data.out, 1)
    pred = logits.max(1)[1]

    print(data.file_path[0])
    create_h5(data.file_path[0],
              pred.cpu().data.numpy(),
              key='cell_predictions', voxel_size=None)

    create_h5(data.file_path[0],
              data.out.cpu().data.numpy(),
              key='cell_net_out', voxel_size=None)
