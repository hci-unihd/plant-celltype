import glob

from plantcelltype.visualization.napari_visualization import CellTypeViewer

data_path = "/home/lcerrone/data/ovules/ovules-celltype-new/419_a_ct.h5"
files = glob.glob(f'{data_path}')

ct_viewer = CellTypeViewer(data_path)
ct_viewer()
