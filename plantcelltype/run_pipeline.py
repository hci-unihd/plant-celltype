import os
import time

from plantcelltype.features.build_features import build_basic_cell_features, build_basic, build_es_proposal
from plantcelltype.features.build_features import build_basic_edges_features, build_edges_points_samples
from plantcelltype.features.build_features import build_cell_points_samples
from plantcelltype.features.build_features import build_edges_planes, build_lrs, build_pca_features
from plantcelltype.features.build_features import build_grs_from_labels, build_naive_grs
from plantcelltype.features.build_features import build_length_along_local_axis, build_cell_dot_features
from plantcelltype.graphnn.predict import run_predictions
from plantcelltype.utils import export_full_stack, open_full_stack
from plantcelltype.utils.io import import_segmentation
from plantcelltype.utils.utils import load_paths
from plantcelltype.visualization.napari_visualization import CellTypeViewer


def preprocessing(config):
    files = load_paths(config['file_list'], filter_h5=False)
    train_data_voxels_size = config.get('train_data_voxels_size', None)
    export_location = config.get('out_dir', None)
    default_seg_key = config.get('default_seg_key', 'segmentation')
    os.makedirs(export_location, exist_ok=True)

    for i, file in enumerate(files):
        timer = - time.time()
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - processing: {file}')

        csv_path = file.replace('.h5', '_annotations.csv')
        csv_path = csv_path if os.path.isfile(csv_path) else None

        base, stack_name = os.path.split(file)
        stack_name, _ = os.path.splitext(stack_name)
        export_location = base if export_location is None else export_location
        assert os.path.isdir(export_location)
        out_file = os.path.join(export_location, f'{stack_name}_ct.h5')

        stack = import_segmentation(file, key=default_seg_key, out_voxel_size=train_data_voxels_size)
        stack = build_basic(stack, csv_path=csv_path)
        stack = build_basic_cell_features(stack)
        stack = build_es_proposal(stack)
        stack = build_cell_points_samples(stack)

        # export processed files
        export_full_stack(out_file, stack)
        timer += time.time()
        print(f'{progress} - runtime: {timer:.2f}s')


def manual_grs(files):
    for i, file in enumerate(files):
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - fix-grs: {file}')
        ct_viewer = CellTypeViewer(file)
        ct_viewer()


def automatic_grs(files, step=build_naive_grs):
    for i, file in enumerate(files):
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - fix-grs: {file}')
        stack, at = open_full_stack(file)
        stack = step(stack, at)
        export_full_stack(file, stack)


def fix_grs(config):
    mode = config.get('mode', 'trivial_grs')
    files = load_paths(config['file_list'], filter_h5=True)

    if mode == 'trivial_grs':
        return automatic_grs(files)
    elif mode == 'label_grs':
        return automatic_grs(files, step=build_grs_from_labels)
    elif mode == 'manual_grs':
        return manual_grs(files)
    else:
        raise NotImplementedError


def advanced_preprocessing(config):
    files = load_paths(config['file_list'], filter_h5=True)

    for i, file in enumerate(files):
        timer = - time.time()
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - advanced-features: {file}')

        stack, at = open_full_stack(file)

        # basics
        stack = build_basic_edges_features(stack, at)
        stack = build_edges_points_samples(stack)

        # lrs
        stack = build_edges_planes(stack, at)
        stack = build_lrs(stack, at)

        # advanced
        stack = build_pca_features(stack, at)
        stack = build_length_along_local_axis(stack, at)
        stack = build_cell_dot_features(stack, at)

        # export processed files
        export_full_stack(file, stack)
        timer += time.time()
        print(f'{progress} - runtime: {timer:.2f}s')


def main(config, process=None):

    if process is None:
        process = {'preprocessing': preprocessing,
                   'grs_step': fix_grs,
                   'advanced_features_step': advanced_preprocessing,
                   'ct_predictions': run_predictions}

    for process_key, process_func in process.items():
        sub_config = config[process_key]
        if sub_config.get('state', False):
            process_func(sub_config)


def process_train_data(config):
    _temp_id = 'XXXX'
    for stack in ['2-III', '2-IV', '2-V', '3-I', '3-II', '3-III', '3-IV', '3-V', '3-VI']:
        config['preprocessing']['file_list'] = config['preprocessing']['file_list'].replace(_temp_id, stack)
        config['preprocessing']['out_dir'] = config['preprocessing']['out_dir'].replace(_temp_id, stack)
        files_list = config['grs_step']['file_list'].replace(_temp_id, stack)

        config['grs_step']['file_list'] = files_list
        config['advanced_features_step']['file_list'] = files_list
        config['ct_predictions']['loader']['file_list'] = files_list

        main(config)
