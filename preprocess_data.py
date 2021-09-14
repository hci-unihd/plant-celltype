import glob
import os
import time

from plantcelltype.utils.io import import_segmentation
from plantcelltype.utils.axis_transforms import AxisTransformer
from plantcelltype.utils import export_full_stack
from plantcelltype.features.build_features import build_basic_cell_features, build_basic, build_es_proposal
from plantcelltype.features.build_features import build_cell_points_samples


train_data_voxels_size = [0.25, 0.2, 0.2]
raw_data_location = "/home/lcerrone/Downloads/tejasvinee/predictions_segmentations/*segmentation.tif"
export_location = "/home/lcerrone/Downloads/"
default_seg_key = "segmentation"
files = glob.glob(f'{raw_data_location}')


for i, file in enumerate(files):
    timer = - time.time()
    progress = f'{i}/{len(files)}'
    print(f'{progress} - processing: {file}')

    base, stack_name = os.path.split(file)
    stack_name, _ = os.path.splitext(stack_name)
    export_location = base if export_location is None else export_location
    assert os.path.isdir(export_location)
    out_file = os.path.join(export_location, f'{stack_name}_ct.h5')

    stack = import_segmentation(file, key=default_seg_key, out_voxel_size=train_data_voxels_size)
    at = AxisTransformer()
    stack = build_basic(stack)
    stack = build_basic_cell_features(stack)
    stack = build_es_proposal(stack)
    stack = build_cell_points_samples(stack)

    # export processed files
    export_full_stack(out_file, stack)
    timer += time.time()
    print(f'{progress} - runtime: {timer:.2f}s')
