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


def export_predictions_as_csv(file_path, cell_ids, cell_predictions, ensemble=None):
    keys = ['label', 'parent_label']
    file_path = file_path.replace('.h5', '.csv')

    out_dict = {}
    with open(file_path, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        for c_id, c_pred in zip(cell_ids, cell_predictions):
            dict_writer.writerow({keys[0]: c_id, keys[1]: c_pred})
            out_dict[c_id] = c_pred
    return out_dict


def export_predictions_as_h5(file_path,
                             celltype_predictions,
                             network_out=None,
                             ensemble=None,
                             default_group='net_predictions'):
    ensemble = '' if ensemble is None else f'_{ensemble}'
    create_h5(file_path,
              celltype_predictions,
              key=f'{default_group}/celltype{ensemble}', voxel_size=None)

    if network_out is not None:
        create_h5(file_path,
                  network_out,
                  key=f'{default_group}/cell_net_out{ensemble}', voxel_size=None)


def compute_predictions(model, data):
    data, _ = model.forward(data)
    logits = torch.log_softmax(data.out, 1)
    cell_predictions = logits.max(1)[1]
    cell_predictions = cell_predictions.cpu().data.numpy().astype('int32')
    return cell_predictions, data.out.cpu().data.numpy()


def run_simple_prediction(config, checkpoint=None, ensemble=None):
    check_point = config['checkpoint'] if checkpoint is None else checkpoint
    check_point_config = f'{check_point}/config.yaml'
    check_point_weights = f'{check_point}/checkpoints/best_class_acc_*ckpt'
    check_point_weights = glob.glob(check_point_weights)[0]
    model_config = load_yaml(check_point_config)

    if config['loader']['mode'] == 'test':
        test_loader, in_features, in_edges_attr = get_test_loaders(config['loader'])
    elif config['loader']['mode'] == 'files':
        test_loader, in_features, in_edges_attr = get_files_loader(config['loader'])
    else:
        raise NotImplementedError

    model = get_model(model_config, in_features=in_features, in_edges_attr=in_edges_attr)
    model = model.load_from_checkpoint(check_point_weights)

    results_dict = {}
    for data in test_loader:
        cell_predictions, net_output = compute_predictions(model, data)

        if config.get('save_h5_predictions', False):
            export_predictions_as_h5(data.file_path,
                                     celltype_predictions=cell_predictions,
                                     network_out=data.out.cpu().data.numpy(),
                                     ensemble=ensemble)

        results_dict[data.file_path] = export_predictions_as_csv(data.file_path,
                                                                 data.node_ids.cpu().data.numpy(),
                                                                 cell_predictions)
    return results_dict


def run_ensemble_prediction(config):
    results = {}
    for i, checkpoint in enumerate(config['checkpoint']):
        _results = run_simple_prediction(config, checkpoint=checkpoint, ensemble=i)
        results[i] = _results


def run_prediction(config):
    checkpoints = config['checkpoint']
    if isinstance(checkpoints, str):
        run_simple_prediction(config)
    elif isinstance(config, list):
        run_ensemble_prediction(config)
