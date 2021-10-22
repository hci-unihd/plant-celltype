import numpy as np
import pandas as pd
from pctg_benchmark.utils.io import load_yaml
import os


def aggregate_macro(score, index=None):
    if index is not None:
        score = np.delete(score, index)
    return np.average(score)


def results_to_dataframe(list_results, **kwargs):
    all_records = []
    for exp_res in list_results:
        records = {'Accuracy': exp_res['results']['accuracy_micro'],
                   'F1': exp_res['results']['f1_micro'],
                   'Accuracy Class': aggregate_macro(exp_res['results']['accuracy_class'], index=7),
                   'F1 Class': aggregate_macro(exp_res['results']['f1_class'], index=7),
                   'file_path': exp_res['file_path'][0],
                   'stack': exp_res['meta']['stack'][0],
                   'stage': exp_res['meta']['stage'][0],
                   'multiplicity': exp_res['meta']['multiplicity'][0],
                   'unique_idx': exp_res['meta']['unique_idx'][0]}

        records.update(kwargs)
        all_records.append(records)
    return pd.DataFrame.from_records(all_records)


def summarize_cross_validation_run(list_checkpoint_path, run_name, save=True):

    # setup directory and path
    save_dir_name = f'{run_name}_summary'
    _sample_checkpoint_path = list_checkpoint_path[0]
    version_base, version = os.path.split(_sample_checkpoint_path)
    base, _ = os.path.split(version_base)
    out_path = os.path.join(base, save_dir_name)

    # create dataframe
    list_results, config = [], {}
    for file_dir in list_checkpoint_path:
        config_path = os.path.join(file_dir, 'experiments.yaml')
        config = load_yaml(config_path)
        config_run = config.pop('run')
        list_results += config_run['results']['val']['results']
    glob_df = results_to_dataframe(list_results, **config)

    if save:
        # save to disc
        with open(os.path.join(out_path, 'summary_readme.txt'), 'w') as f:
            f.write('Experiment source\n')
            for file_dir in list_checkpoint_path:
                f.write(f'{file_dir}\n')

        os.makedirs(out_path, exist_ok=True)
        glob_df.to_pickle(os.path.join(out_path, 'results_df.pkl'))
    return glob_df
