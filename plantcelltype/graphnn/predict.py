import csv
import glob

import torch

from plantcelltype.graphnn.trainer import get_model
from plantcelltype.utils import create_h5
from pctg_benchmark.utils.io import load_yaml
from plantcelltype.utils.utils import load_paths
from plantcelltype.graphnn.trainer import datasets
from torch_geometric.loader import DataLoader
from pctg_benchmark.loaders.build_dataset import default_build_torch_geometric_data


def get_test_loaders(config):
    test_dataset = datasets[config['mode']](**config['val_dataset'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config['val_batch_size'],
                             num_workers=config['num_workers'],
                             shuffle=False)

    in_edges_attr = test_loader.in_edges_attr
    in_features = test_dataset.in_features
    return test_loader, in_features, in_edges_attr


def get_files_loader(config):
    paths = load_paths(config['files_list'])

    dataset_config = config.get('dataset', None)
    meta = config.get('meta', None)
    all_data, data = [], None
    for file in paths:
        data, _ = default_build_torch_geometric_data(file,
                                                     config=dataset_config,
                                                     meta=meta)
        all_data.append(data)

    in_edges_attr = data.in_edges_attr
    in_features = data.in_features
    return all_data, in_features, in_edges_attr


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
    check_point_config = f'{check_point}/experiments.yaml'
    check_point_weights = f'{check_point}/checkpoints/*ckpt'
    check_point_weights = glob.glob(check_point_weights)[0]
    model_config = load_yaml(check_point_config)

    if config['loader']['mode'] == 'test':
        test_loader, in_features, in_edges_attr = get_test_loaders(config['loader'])
    elif config['loader']['mode'] == 'files':
        test_loader, in_features, in_edges_attr = get_files_loader(config['loader'])
    else:
        raise NotImplementedError

    save_h5_predictions = config.get('save_h5_predictions', False)

    model = get_model(model_config, in_features=in_features, in_edges_attr=in_edges_attr)
    model = model.load_from_checkpoint(check_point_weights)

    for data in test_loader:
        data, _ = model.forward(data)
        logits = torch.log_softmax(data.out, 1)
        cell_predictions = logits.max(1)[1]
        cell_predictions = cell_predictions.cpu().data.numpy().astype('int32')

        if save_h5_predictions:
            create_h5(data.file_path,
                      cell_predictions,
                      key='cell_predictions', voxel_size=None)

            create_h5(data.file_path,
                      data.out.cpu().data.numpy(),
                      key='cell_net_out', voxel_size=None)

        export_predictions_as_csv(data.file_path,
                                  data.node_ids.cpu().data.numpy(),
                                  cell_predictions)
