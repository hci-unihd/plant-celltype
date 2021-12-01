import csv
import glob

import torch

import numpy as np
import pandas as pd
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

    out_dict = {}
    with open(file_path, "w") as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        for c_id, c_pred in zip(cell_ids, cell_predictions):
            dict_writer.writerow({keys[0]: c_id, keys[1]: c_pred})
            out_dict[c_id] = c_pred
    return out_dict


def create_df(cell_ids, cell_predictions, ensemble=None):
    label_key, prediction_key = 'label', 'parent_label_model'
    prediction_key = f'{prediction_key}' if ensemble is None else f'{prediction_key}_{ensemble}'
    df_dict = {label_key: cell_ids, prediction_key: cell_predictions}
    return pd.DataFrame.from_dict(df_dict)


def summarize_df(predictions_df: pd.DataFrame):
    label_key, prediction_key, confidence_key = 'label', 'parent_label', 'confidence_key'
    keys = list(predictions_df.keys())
    keys.remove('label')

    cell_ids = list(predictions_df['label'])

    parent_label, confidence_label = [], []
    for i, _x in predictions_df[keys].iterrows():
        pred, pred_counts = np.unique(_x.to_numpy(), return_counts=True)
        argmax_pred = np.argmax(pred_counts)
        max_pred, confidence = pred[argmax_pred], pred_counts[argmax_pred] / np.sum(pred_counts)
        confidence = round(confidence, 2)
        parent_label.append(max_pred)
        confidence_label.append(confidence)

    return pd.DataFrame.from_dict({label_key: cell_ids,
                                   prediction_key: parent_label,
                                   confidence_key: confidence_label})


def create_csv(file_path, predictions_df: pd.DataFrame, summarize=False):
    file_path = file_path.replace('.h5', '.csv')
    if summarize:
        predictions_df = summarize_df(predictions_df)
    predictions_df.to_csv(file_path, index=False)


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

        results_dict[data.file_path] = create_df(data.node_ids.cpu().data.numpy(),
                                                 cell_predictions,
                                                 ensemble=ensemble)

    return results_dict


def run_ensemble_prediction(config):
    results_list = []
    for i, checkpoint in enumerate(config['checkpoint']):
        results_pd = run_simple_prediction(config, checkpoint=checkpoint, ensemble=i)
        results_list.append(results_pd)

    for key in results_list[0].keys():
        results_pd = pd.concat([_df[key] for _df in results_list], axis=1)
        create_csv(file_path=key, predictions_df=results_pd, summarize=True)


def run_prediction(config):
    checkpoints = config['checkpoint']
    if isinstance(checkpoints, str):
        results_dict = run_simple_prediction(config)
        [create_csv(file_path=key, predictions_df=results_dict[key]) for key in results_dict.keys()]

    elif isinstance(config, list):
        run_ensemble_prediction(config)
    else:
        raise NotImplementedError
