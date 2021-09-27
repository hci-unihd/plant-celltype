import copy
import os
import time

from plantcelltype.features.build_features import build_basic_cell_features, build_basic, build_es_proposal
from plantcelltype.features.build_features import build_basic_edges_features, build_edges_points_samples
from plantcelltype.features.build_features import build_cell_points_samples, build_es_features
from plantcelltype.features.build_features import build_edges_planes, build_lrs, build_pca_features
from plantcelltype.features.build_features import build_grs_from_labels_funiculum, build_grs_from_labels_surface
from plantcelltype.features.build_features import build_length_along_local_axis, build_cell_dot_features
from plantcelltype.features.build_features import build_preprocessing
from plantcelltype.features.build_features import build_trivial_grs, build_es_trivial_grs, build_es_pca_grs
from plantcelltype.graphnn.predict import run_predictions
from plantcelltype.utils import export_full_stack, open_full_stack
from plantcelltype.utils.io import import_segmentation, load_axis_transformer
from plantcelltype.utils.utils import load_paths
from plantcelltype.visualization.napari_visualization import CellTypeViewer


def preprocessing(config):
    files = load_paths(config['file_list'], filter_h5=False)
    train_data_voxels_size = config.get('train_data_voxels_size', None)
    export_location = config.get('out_dir', None)
    default_seg_key = config.get('default_seg_key', 'segmentation')
    if export_location is not None:
        os.makedirs(export_location, exist_ok=True)

    for i, file in enumerate(files):
        timer = - time.time()
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - processing: {file}')

        base, stack_name = os.path.split(file)
        stack_name, ext = os.path.splitext(stack_name)
        export_location = base if export_location is None else export_location
        assert os.path.isdir(export_location)

        csv_path = file.replace(ext, '_annotations.csv')
        csv_path = csv_path if os.path.isfile(csv_path) else None

        out_file = os.path.join(export_location, f'{stack_name}_ct.h5')

        stack = import_segmentation(file, key=default_seg_key, out_voxel_size=train_data_voxels_size)

        stack = build_preprocessing(stack)
        stack = build_basic(stack, csv_path=csv_path)

        # stack = build_basic_cell_features(stack)
        stack = build_es_proposal(stack)
        stack = build_es_features(stack)
        stack = build_cell_points_samples(stack)
        stack = build_trivial_grs(stack, load_axis_transformer(stack['attributes']))

        # export processed files
        export_full_stack(out_file, stack)
        timer += time.time()
        print(f'{progress} - runtime: {timer:.2f}s')


from line_profiler import LineProfiler
profile_pre = LineProfiler()
preprocessing = profile_pre(preprocessing)


def manual_grs(files):
    for i, file in enumerate(files):
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - fix-grs: {file}')
        ct_viewer = CellTypeViewer(file)
        ct_viewer()


def automatic_grs(files, step=build_trivial_grs):
    for i, file in enumerate(files):
        progress = f'{i+1}/{len(files)}'
        print(f'{progress} - fix-grs: {file}')
        stack, at = open_full_stack(file)
        at.reset_axis()
        stack = step(stack, at)
        export_full_stack(file, stack)


def fix_grs(config):
    mode = config.get('mode', 'trivial_grs')
    files = load_paths(config['file_list'], filter_h5=True)

    if mode == 'trivial_grs':
        return automatic_grs(files)

    elif mode == 'es_trivial_grs':
        return automatic_grs(files, step=build_es_trivial_grs)

    elif mode == 'es_pca_grs':
        return automatic_grs(files, step=build_es_pca_grs)

    elif mode == 'label_grs_funiculum':
        return automatic_grs(files, step=build_grs_from_labels_funiculum)

    elif mode == 'label_grs_surface':
        return automatic_grs(files, step=build_grs_from_labels_surface)

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
    print(profile_pre.print_stats())


def process_train_data(config):
    _temp_id = 'XXXX'
    for stack in ['2-III', '2-IV', '2-V', '3-I', '3-II', '3-III', '3-IV', '3-V', '3-VI']:
        _config = copy.deepcopy(config)
        _config['preprocessing']['file_list'] = _config['preprocessing']['file_list'].replace(_temp_id, stack)
        _config['preprocessing']['out_dir'] = _config['preprocessing']['out_dir'].replace(_temp_id, stack)
        files_list = _config['grs_step']['file_list'].replace(_temp_id, stack)

        _config['grs_step']['file_list'] = files_list
        _config['advanced_features_step']['file_list'] = files_list
        _config['ct_predictions']['loader']['file_list'] = files_list

        main(_config)
