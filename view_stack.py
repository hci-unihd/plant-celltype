from plantcelltype.visualization.napari_visualization import CellTypeViewer

if __name__ == '__main__':
    data_path = "/home/lcerrone/data/ovules/ovules-celltype-new/es_pca_grs/late_cropped_ds3/2-III/294_D_a_ct.h5"
    ct_viewer = CellTypeViewer(data_path)
    ct_viewer()
