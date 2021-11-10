import os

import pandas as pd

from pctg_benchmark.evaluation.metrics import aggregate_class
from pctg_benchmark.utils.io import load_yaml


def expand_class_metric(score, name='metric'):
    return {f'{name} - {i}': s for i, s in enumerate(score)}


def results_to_dataframe(list_results, index_to_ignore=None, **kwargs):
    all_records = []
    for exp_res in list_results:
        accuracy_class, num_nan = aggregate_class(exp_res['results']['accuracy_class'], index=index_to_ignore)
        accuracy_f1, _ = aggregate_class(exp_res['results']['f1_class'], index=index_to_ignore)

        records = {'Accuracy': exp_res['results']['accuracy_micro'],
                   'F1': exp_res['results']['f1_micro'],
                   'Dice': exp_res['results']['dice'],
                   'Accuracy Class': accuracy_class,
                   'F1 Class': accuracy_f1,
                   'num Nan': num_nan,
                   'file_path': exp_res['file_path'][0],
                   'stack': exp_res['meta']['stack'][0],
                   'stage': exp_res['meta']['stage'][0],
                   'multiplicity': exp_res['meta']['multiplicity'][0],
                   'unique_idx': exp_res['meta']['unique_idx'][0]}

        records.update(expand_class_metric(exp_res['results']['accuracy_class'], name='Accuracy Class'))
        records.update(expand_class_metric(exp_res['results']['f1_class'], name='F1 Class'))

        records.update(kwargs)
        all_records.append(records)
    return pd.DataFrame.from_records(all_records)


def summarize_cross_validation_run(list_checkpoint_path, save=True):

    # setup directory and path
    save_dir_name = f'summary'
    _sample_checkpoint_path = list_checkpoint_path[0]
    version_base, version = os.path.split(_sample_checkpoint_path)
    base, _ = os.path.split(version_base)
    out_path = os.path.join(base, save_dir_name)

    # create dataframe
    list_results, config, index_to_ignore = [], {}, None
    for file_dir in list_checkpoint_path:
        config_path = os.path.join(file_dir, 'config.yaml')
        config = load_yaml(config_path)
        mode = config.get('mode')
        index_to_ignore = None if mode == 'EdgesClassification' else 7
        config_run = config.pop('run')
        list_results += config_run['results']['val']['results']
    glob_df = results_to_dataframe(list_results, index_to_ignore=index_to_ignore, **config)

    if save:
        # save to disc
        os.makedirs(out_path, exist_ok=True)
        with open(os.path.join(out_path, 'summary_readme.txt'), 'w') as f:
            f.write('Experiment source\n')
            for file_dir in list_checkpoint_path:
                f.write(f'{file_dir}\n')
        glob_df.to_pickle(os.path.join(out_path, 'results_df.pkl'))
    return glob_df
