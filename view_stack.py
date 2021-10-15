import argparse
import os

from plantcelltype.visualization.napari_visualization import CellTypeViewer


def parser():
    _parser = argparse.ArgumentParser(description='plant-celltype training experiments')
    _parser.add_argument('--stack', type=str, help='Path to the CellType h5 stack', required=True)
    args = _parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parser()

    minimal_keys = ('rw_centrality',
                    'degree_centrality',
                    'volume_voxels',
                    'surface_voxels',
                    'bg_edt_um',
                    'hops_to_bg')
    #minimal_keys = []

    if os.path.isfile(_args.stack):
        ct_viewer = CellTypeViewer(_args.stack, features=minimal_keys)
        ct_viewer()
    else:
        print(_args.stack, ' does not exists.')
