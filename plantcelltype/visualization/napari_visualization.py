import napari
import numpy as np

from plantcelltype.utils import map_edges_features2rag_boundaries, map_cell_features2segmentation
from plantcelltype.graphnn.data_loader import gt_mapping_wb


def create_prediction_label_image(stack):
    cell_labels = stack['cell_labels']
    cell_predictions = stack['cell_predictions']

    for i in range(cell_labels.shape[0]):
        # map labels for consistency
        cell_labels[i] = gt_mapping_wb[cell_labels[i]] + 1
        cell_predictions[i] += 1

    predictions = map_cell_features2segmentation(stack['segmentation'], stack['cell_ids'], cell_predictions)
    labels = map_cell_features2segmentation(stack['segmentation'], stack['cell_ids'], cell_labels)
    predictions, labels = predictions.astype('uint16'), labels.astype('uint16')
    return predictions, labels


def visualize_all_cell_features(stack, print_features=True):
    viewer = napari.Viewer()
    viewer.add_labels(stack['segmentation'],
                      name='segmentation',
                      scale=stack['attributes']['element_size_um'])

    if 'cell_predictions' in stack:
        predictions, labels = create_prediction_label_image(stack)

        viewer.add_labels(labels, name='labels', scale=stack['attributes']['element_size_um'])
        viewer.add_labels(predictions, name='predictions', scale=stack['attributes']['element_size_um'])
        viewer.add_labels(np.where(predictions != labels, 19, 0),
                          name='errors',
                          scale=stack['attributes']['element_size_um'])

    elif 'labels' in stack:
        viewer.add_labels(stack['labels'].astype('uint16'),
                          name='labels',
                          scale=stack['attributes']['element_size_um'])

    if print_features and 'cell_features' in stack:
        print("cell features:")
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

    if 'es_com_voxels' not in stack['attributes']:
        stack['attributes']['es_com_voxels'] = np.array([])

    viewer.add_points(stack['attributes']['es_com_voxels'] * stack['attributes']['element_size_um'],
                      name='es',
                      n_dimensional=True,
                      size=5)

    viewer.add_points([],
                      ndim=3,
                      name='Main Symmetry Axis',
                      n_dimensional=True,
                      face_color='green',
                      size=2)
    viewer.add_points([],
                      ndim=3,
                      name='Secondary Symmetry Axis ',
                      n_dimensional=True,
                      face_color='red',
                      size=2)

    return viewer


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