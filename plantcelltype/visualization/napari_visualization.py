import napari
import numpy as np

from plantcelltype.utils import map_edges_features2rag_boundaries, map_cell_features2segmentation
from plantcelltype.graphnn.data_loader import gt_mapping_wb


def visualize_all_cell_features(stack):
    viewer = napari.Viewer()
    viewer.add_labels(stack['segmentation'],
                      name='segmentation',
                      scale=stack['attributes']['element_size_um'])

    if 'labels' in stack:
        viewer.add_labels(stack['labels'],
                          name='labels',
                          scale=stack['attributes']['element_size_um'])

    if 'cell_predictions' in stack:
        cell_labels = stack['cell_labels']
        cell_predictions = stack['cell_predictions']
        for i in range(cell_labels.shape[0]):
            cell_labels[i] = gt_mapping_wb[cell_labels[i]] + 1
            cell_predictions[i] += 1

        feat = map_cell_features2segmentation(stack['segmentation'], stack['cell_ids'], cell_predictions)
        labels_2 = map_cell_features2segmentation(stack['segmentation'], stack['cell_ids'], cell_labels)

        viewer.add_labels(labels_2, name='labels_t', scale=stack['attributes']['element_size_um'])
        viewer.add_labels(feat.astype('uint16'), name='predictions', scale=stack['attributes']['element_size_um'])
        viewer.add_labels(np.where(feat.astype('uint16') != labels_2, 19, 0),
                          name='difference',
                          scale=stack['attributes']['element_size_um'])

    print("cell features:")
    if 'cell_features' in stack:
        for key, value in stack['cell_features'].items():
            if isinstance(value, np.ndarray):
                print(key, value.shape, value.dtype)
                if value.ndim == 1:
                    feat = map_cell_features2segmentation(stack['segmentation'], stack['cell_ids'], value)
                    viewer.add_image(feat,
                                     name=key,
                                     colormap='inferno',
                                     visible=False,
                                     scale=stack['attributes']['element_size_um'])


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