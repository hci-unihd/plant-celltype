import glob
import os
from plantcelltype.utils import open_full_stack, export_full_stack
from plantcelltype.features.build_features import build_basic_cell_features, build_basic_edges_features, build_basic

data_location = "./data/ovules-celltype/late_cropped_ds3/"
out_location = "./data/ovules-celltype-processed/"

files = glob.glob(f'{data_location}/**/*.h5')
for i, h5_file in enumerate(files):
    print(f'{i}/{len(files)} - processing: {h5_file} ')
    csv_file = h5_file.replace(".h5", "_annotations.csv")
    _base, stack_name = os.path.split(h5_file)
    _, stack_dir = os.path.split(_base)
    out_dir = os.path.join(out_location, stack_dir)
    out_path = os.path.join(out_dir, stack_name)
    os.makedirs(out_dir, exist_ok=True)

    # start processing
    stack, at = open_full_stack(h5_file)

    stack = build_basic(stack, csv_path=csv_file)
    stack = build_basic_cell_features(stack)
    stack = build_basic_edges_features(stack, at)

    # export processed files
    export_full_stack(out_path, stack)
