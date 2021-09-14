import glob
import time

from plantcelltype.utils import export_full_stack, open_full_stack
from plantcelltype.features.build_features import build_basic_edges_features, build_edges_points_samples
from plantcelltype.features.build_features import build_edges_planes, build_lrs, build_pca_features
from plantcelltype.features.build_features import build_length_along_local_axis, build_cell_dot_features


raw_data_location = "/home/lcerrone/data/ovules/ovules-celltype-new/*.h5"
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
