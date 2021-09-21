import copy

import napari
import numpy as np
from scipy.ndimage import zoom
from skspatial.objects import Line, Vector

from plantcelltype.graphnn.data_loader import gt_mapping_wb
from plantcelltype.utils import map_edges_features2rag_boundaries, map_cell_features2segmentation
from plantcelltype.utils.io import open_full_stack, export_full_stack


def create_prediction_label_image(stack, segmentation=None):
    segmentation = segmentation if segmentation is None else stack['segmentation']
    cell_labels = copy.deepcopy(stack['cell_labels'])
    cell_predictions = stack['cell_predictions']
    for i in range(cell_labels.shape[0]):
        # map labels for consistency
        cell_labels[i] = gt_mapping_wb[cell_labels[i]] + 1
        cell_predictions[i] += 1

    predictions = map_cell_features2segmentation(segmentation, stack['cell_ids'], cell_predictions)
    labels = map_cell_features2segmentation(segmentation, stack['cell_ids'], cell_labels)
    predictions, labels = predictions.astype('uint16'), labels.astype('uint16')
    return predictions, labels


class CellTypeViewer:
    def __init__(self, path, features=None, scale=(1, 1, 1)):
        self.path = path
        stack, at = open_full_stack(path)
        self.segmentation = stack['segmentation']
        self.voxel_size = stack['attributes']['element_size_um']
        self.update_scaling(scale)

        self.stack = stack
        self.at = at
        if 'cell_features' in stack:
            self.view_features = stack['cell_features'].keys() if features is None else features

    def update_scaling(self, scale=(1, 1, 1)):
        if np.prod(scale) != 1:
            self.segmentation = zoom(self.segmentation, scale, order=0)
        self.voxel_size *= np.array(scale)

    def __call__(self):
        viewer = napari.Viewer(title='Cell Features')
        viewer.add_labels(self.segmentation,
                          name='segmentation',
                          scale=self.voxel_size)

        if 'cell_predictions' in self.stack:
            predictions, labels = create_prediction_label_image(self.stack, self.segmentation)
            viewer.add_labels(labels, name='labels', scale=self.voxel_size, visible=False)
            viewer.add_labels(predictions, name='predictions', scale=self.voxel_size)
            viewer.add_labels(np.where(predictions != labels, 19, 0),
                              visible=False,
                              name='errors',
                              scale=self.stack['attributes']['element_size_um'])
        else:
            labels = map_cell_features2segmentation(self.segmentation,
                                                    self.stack['cell_ids'],
                                                    self.stack['cell_labels'])
            viewer.add_labels(labels, name='labels', scale=self.voxel_size)

        if 'cell_features' in self.stack:
            cell_features = self.stack['cell_features']
            for key in self.view_features:
                value = cell_features[key]
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        feat = map_cell_features2segmentation(self.segmentation,
                                                              self.stack['cell_ids'],
                                                              value)
                        viewer.add_image(feat,
                                         name=key,
                                         colormap='inferno',
                                         visible=False,
                                         scale=self.voxel_size)

        grs_axis, main_axis_points, secondary_axis_points = self.build_grs_vectors()
        viewer.add_vectors(grs_axis, name='current-grs', length=10, edge_color='blue', edge_width=1)

        viewer.add_points(self.stack['attributes']['global_reference_system_origin'],
                          name='grs origin',
                          n_dimensional=True,
                          size=5)

        viewer.add_points(main_axis_points,
                          ndim=3,
                          name='grs 1st axis',
                          n_dimensional=True,
                          face_color='green',
                          size=2)

        viewer.add_points(secondary_axis_points,
                          ndim=3,
                          name='grs 2nd axis',
                          n_dimensional=True,
                          face_color='red',
                          size=2)

        @viewer.bind_key('U')
        def _update(_viewer):
            """Update axis"""
            self.update_grs()

        self.viewer = viewer
        napari.run()

    def update_grs(self):
        fu_points = self.viewer.layers[-1].data
        main_ax_points = self.viewer.layers[-2].data
        origin_points = self.viewer.layers[-3].data

        # update labels
        list_origin_idx = []
        for point in origin_points:
            point = (np.array(point) / self.voxel_size).astype('int32')

            cell_idx = self.segmentation[point[0], point[1], point[2]]
            list_origin_idx.append(cell_idx)
            self.stack['segmentation'] = np.where(self.segmentation == cell_idx,
                                                  list_origin_idx[0],
                                                  self.segmentation)

        # main axis
        new_origin_points = np.mean(np.where(self.stack['segmentation'] == list_origin_idx[0]), 1)
        main_ax_points = [(np.array(point) / self.voxel_size).astype('int32') for point in main_ax_points]
        main_ax_points.append(new_origin_points.tolist())
        line = Line.best_fit(main_ax_points)
        main_axis = line.vector

        # second axis
        fu_points = fu_points[0]
        proj_secondary_axis = line.project_point(fu_points)
        second_axis = (Vector(proj_secondary_axis) - Vector(fu_points)).unit()

        # third axis
        third_axis = main_axis.cross(second_axis)

        self.stack['attributes']['global_reference_system_origin'] = new_origin_points
        self.stack['attributes']['global_reference_system_axis'] = (main_axis, second_axis, third_axis)
        export_full_stack(self.path, self.stack)
        print("stack exported")

    def build_grs_vectors(self):
        axis = self.stack['attributes']['global_reference_system_axis']
        origin_point = self.stack['attributes']['global_reference_system_origin']
        vector = np.zeros((3, 2, 3))

        main_points = [origin_point + 10 * axis[0], origin_point - 10 * axis[0]]
        secondary_points = [origin_point + 10 * axis[1], origin_point - 10 * axis[1]]

        vector[0, 0] = origin_point
        vector[1, 0] = origin_point
        vector[2, 0] = origin_point

        vector[0, 1] = axis[0]
        vector[1, 1] = axis[1]
        vector[2, 1] = axis[2]
        return vector, main_points, secondary_points


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
