import napari
import numpy as np
from scipy.ndimage import zoom
from sklearn.decomposition import PCA

from plantcelltype.utils import map_edges_features2rag_boundaries, map_cell_features2segmentation
from plantcelltype.graphnn.data_loader import gt_mapping_wb
from skspatial.objects import Line, Vector
from plantcelltype.utils.io import open_full_stack, export_full_stack


def create_prediction_label_image(stack, segmentation=None):
    segmentation = segmentation if segmentation is None else stack['segmentation']
    cell_labels = stack['cell_labels']
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
    def __init__(self, path, view_features=True, scale=(1, 1, 1)):
        self.path = path
        stack, _ = open_full_stack(path)
        self.segmentation = stack['segmentation']
        self.voxel_size = stack['attributes']['element_size_um']
        self.update_scaling(scale)

        self.stack = stack
        self.view_features = view_features

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

            viewer.add_labels(labels, name='labels', scale=self.voxel_size)
            viewer.add_labels(predictions, name='predictions', scale=self.voxel_size)
            viewer.add_labels(np.where(predictions != labels, 19, 0),
                              name='errors',
                              scale=self.stack['attributes']['element_size_um'])
        else:
            labels = map_cell_features2segmentation(self.segmentation,
                                                    self.stack['cell_ids'],
                                                    self.stack['cell_labels'])
            viewer.add_labels(labels, name='labels', scale=self.voxel_size)

        if self.view_features and 'cell_features' in self.stack:
            for key, value in self.stack['cell_features'].items():
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

        if 'es_com_voxels' not in self.stack['attributes']:
            self.stack['attributes']['es_com_voxels'] = np.array([])

        main_vector, _ = self.compute_pca_es()
        viewer.add_vectors(main_vector, length=10, edge_color='green', edge_width=2)

        viewer.add_points(self.stack['attributes']['es_com_voxels'] * self.voxel_size,
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
                          name='Secondary Funiculum Axis',
                          n_dimensional=True,
                          face_color='red',
                          size=2)

        @viewer.bind_key('U')
        def _update(_viewer):
            """Update axis"""
            self.update_grs()

        self.viewer = viewer

    def update_grs(self):
        fu_points = self.viewer.layers[-1].data
        main_ax_points = self.viewer.layers[-2].data
        es_points = self.viewer.layers[-3].data

        # update labels
        list_es_idx = []
        for point in es_points:
            point = (np.array(point) / self.voxel_size).astype('int32')

            cell_idx = self.segmentation[point[0], point[1], point[2]]
            list_es_idx.append(cell_idx)
            self.stack['segmentation'] = np.where(self.segmentation == cell_idx,
                                                  list_es_idx[0],
                                                  self.segmentation)

        # main axis
        new_es_points = np.mean(np.where(self.stack['segmentation'] == list_es_idx[0]), 1)
        main_ax_points = [(np.array(point) / self.voxel_size).astype('int32') for point in main_ax_points]
        main_ax_points.append(new_es_points.tolist())
        line = Line.best_fit(main_ax_points)
        main_axis = line.vector

        # second axis
        fu_points = fu_points[0]
        proj_secondary_axis = line.project_point(fu_points)
        second_axis = (Vector(proj_secondary_axis) - Vector(fu_points)).unit()

        # third axis
        third_axis = main_axis.cross(second_axis)

        self.stack['attributes']['global_reference_system_origin'] = new_es_points
        self.stack['attributes']['global_reference_system_axis'] = (main_axis, second_axis, third_axis)
        export_full_stack(self.path, self.stack)
        print("stack exported")

    def compute_pca_es(self):
        es_idx = self.stack['attributes']['es_index'][0]
        es_samples = self.stack['cell_samples']['random_samples'][es_idx]
        es_samples = es_samples.astype('float32')
        es_samples *= self.voxel_size[None, :]
        pca = PCA()
        pca.fit(es_samples)

        es_com = self.stack['attributes']['es_com_voxels'] * self.voxel_size
        main_vector = self._build_vector(es_com, pca.components_[0])
        secondary_vector = self._build_vector(es_com, pca.components_[1])
        return main_vector, secondary_vector

    @staticmethod
    def _build_vector(es_com, component):
        vector = np.zeros((2, 2, 3))
        vector[0, 0] = es_com
        vector[1, 0] = es_com
        vector[0, 1] = component
        vector[1, 1] = - component
        return vector


def visualize_all_cell_features(stack, print_features=True, scale=(1, 1, 1)):
    segmentation = stack['segmentation']
    if np.prod(scale) == 1:
        segmentation = zoom(segmentation, scale, order=0)

    viewer = napari.Viewer(title='Cell Features')
    viewer.add_labels(segmentation,
                      name='segmentation',
                      scale=stack['attributes']['element_size_um'])

    if 'cell_predictions' in stack:
        predictions, labels = create_prediction_label_image(stack, segmentation)

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
        print('cell features:')
        for key, value in stack['cell_features'].items():
            if isinstance(value, np.ndarray):
                print(key, value.shape, value.dtype)
                if value.ndim == 1:
                    feat = map_cell_features2segmentation(segmentation,
                                                          stack['cell_ids'],
                                                          value)
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
                      name='Secondary Funiculus Axis',
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