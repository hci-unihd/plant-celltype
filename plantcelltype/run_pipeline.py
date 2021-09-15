import glob
import os
import time

from plantcelltype.features.build_features import build_basic_cell_features, build_basic, build_es_proposal
from plantcelltype.features.build_features import build_basic_edges_features, build_edges_points_samples
from plantcelltype.features.build_features import build_cell_points_samples
from plantcelltype.features.build_features import build_edges_planes, build_lrs, build_pca_features
from plantcelltype.features.build_features import build_length_along_local_axis, build_cell_dot_features
from plantcelltype.features.build_features import build_grs_from_labels, build_naive_grs
from plantcelltype.utils import export_full_stack, open_full_stack
from plantcelltype.utils.io import import_segmentation
from plantcelltype.visualization.napari_visualization import CellTypeViewer
from plantcelltype.graphnn.predict import run_predictions
from plantcelltype.utils.io import load_yaml


def preprocessing(config):
    train_data_voxels_size = config.get('train_data_voxels_size', None)
    raw_data_location = config['file_list']
    export_location = config.get('out_dir', None)
    default_seg_key = config.get('default_seg_key', 'segmentation')
    os.makedirs(export_location, exist_ok=True)
    files = glob.glob(f'{raw_data_location}')

    for i, file in enumerate(files):
        timer = - time.time()
        progress = f'{i}/{len(files)}'
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


def manual_grs(config):
    raw_data_location = config['file_list']
    files = glob.glob(f'{raw_data_location}')
    for file in files:
        ct_viewer = CellTypeViewer(file)
        ct_viewer()


def automatic_grs(config, step=build_naive_grs):
    raw_data_location = config['file_list']
    files = glob.glob(f'{raw_data_location}')
    for file in files:
        stack, at = open_full_stack(file)
        stack = step(stack)
        export_full_stack(file, stack)


def trivial_grs(config):
    automatic_grs(config)


def label_grs(config):
    automatic_grs(config, step=build_grs_from_labels)


def advanced_preprocessing(config):
    raw_data_location = config['file_list']
    files = glob.glob(f'{raw_data_location}')

    for i, file in enumerate(files):
        timer = - time.time()
        progress = f'{i}/{len(files)}'
        print(f'{progress} - processing: {file}')

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


def main(config_path, process=None):
    config = load_yaml(config_path)

    if process is None:
        process = {'preprocessing': preprocessing,
                   'grs_step': manual_grs,
                   'advanced_features_step': advanced_preprocessing,
                   'ct_predictions': run_predictions}

    for process_key, process_func in process.items():
        sub_config = config[process_key]
        if sub_config.get('stage', False):
            process_func(sub_config)


def build_train_data(config_path):
    main(config_path, {'preprocessing': preprocessing,
                       'grs_step': label_grs,
                       'advanced_features_step': advanced_preprocessing})
