import numpy as np
import napari
from plantcelltype.utils import map_edges_features2rag_boundaries, map_cell_features2segmentation


def visualize_all_cell_features(stack):
    viewer = napari.Viewer()
    viewer.add_labels(stack['segmentation'], name='segmentation')
    viewer.add_labels(stack['labels'], name='labels')

    print("cell features:")
    for key, value in stack['cell_features'].items():
        if isinstance(value, np.ndarray):
            print(key, value.shape, value.dtype)
            if value.ndim == 1:
                feat = map_cell_features2segmentation(stack['segmentation'], stack['cell_ids'], value)
                viewer.add_image(feat, name=key, colormap='inferno', visible=False)


def visualize_all_edges_features(stack):
    viewer = napari.Viewer()
    viewer.add_labels(stack['rag_boundaries'], name='rag_boundaries')

    print("edges features:")
    for key, value in stack['edges_features'].items():
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                print(key, value.shape, value.dtype)
                feat = map_edges_features2rag_boundaries(stack['rag_boundaries'],
                                                         stack['edges_ids'],
                                                         value)
                viewer.add_image(feat, name=key, colormap='inferno', visible=False)