import csv
import glob

import torch

from plantcelltype.graphnn.trainer import get_model
from plantcelltype.utils import create_h5
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.utils.utils import load_paths


def build_test_loader(config, glob_paths=True):
    if glob_paths:
        config['files_list'] = load_paths(config['files_list'])

    return create_loaders(**config)


def export_predictions_as_csv(file_path, cell_ids, cell_predictions):
    keys = ['label', 'parent_label']
    file_path = file_path.replace('.h5', '.csv')
    with open(file_path, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        for c_id, c_pred in zip(cell_ids, cell_predictions):
            dict_writer.writerow({keys[0]: c_id, keys[1]: c_pred})


def run_predictions(config):
    check_point = config['checkpoint']
    check_point_config = f'{check_point}/config.yaml'
    check_point_weights = f'{check_point}/checkpoints/*ckpt'
    check_point_weights = glob.glob(check_point_weights)[0]
    model_config = load_yaml(check_point_config)

    test_loader = build_test_loader(config['loader'])
    model = get_model(model_config)
    model = model.load_from_checkpoint(check_point_weights)

    for data in test_loader:
        data, _ = model.forward(data)
        logits = torch.log_softmax(data.out, 1)
        cell_predictions = logits.max(1)[1]
        cell_predictions = cell_predictions.cpu().data.numpy().astype('int32')

        create_h5(data.file_path[0],
                  cell_predictions,
                  key='cell_predictions', voxel_size=None)

        create_h5(data.file_path[0],
                  data.out.cpu().data.numpy(),
                  key='cell_net_out', voxel_size=None)

        export_predictions_as_csv(data.file_path[0],
                                  data.cell_ids[0],
                                  cell_predictions)
