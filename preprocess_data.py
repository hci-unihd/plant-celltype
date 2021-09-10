import glob
import os

from plantcelltype.features.build_features import build_basic_cell_features, build_basic_edges_features, build_basic
from plantcelltype.utils import open_full_stack, export_full_stack
from plantcelltype.utils.io import smart_load

raw_data_location = "/home/"
export_location = "./data/ovules-celltype-processed/"
files = glob.glob(f'{raw_data_location}/*')

for i, file in enumerate(files):
    print(f'{i}/{len(files)} - processing: {file} ')
    _base, stack_name = os.path.split(file)
    print(_base, stack_name)
