from plantcelltype.visualization.napari_visualization import CellTypeViewer

if __name__ == '__main__':
    data_path = "/home/lcerrone/data/ovules/ovules-celltype-new/3-III/429_a_ct.h5"
    ct_viewer = CellTypeViewer(data_path)
    ct_viewer()
