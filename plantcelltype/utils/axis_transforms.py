import numpy as np
from skspatial.objects import Plane, Line, Point, Vector
from skspatial.transformation import transform_coordinates


def scale_points(points_coo, voxel_size, reverse=False):
    assert len(voxel_size) == 3
    voxel_size = np.array(voxel_size)
    voxel_size = 1 / voxel_size if reverse else voxel_size
    return points_coo * voxel_size


def transform_coord(points_coo, axis, center=(0, 0, 0), voxel_size=(1, 1, 1)):
    scaled_points = scale_points(points_coo, voxel_size)
    return transform_coordinates(scaled_points, center, axis)


def inv_transform_coord(points_coo, axis, center=(0, 0, 0), voxel_size=(1, 1, 1)):
    center = np.array(center)
    inv_rot_points = transform_coordinates(points_coo, (0, 0, 0), np.linalg.inv(axis))
    inv_rot_centering_points = transform_coordinates(inv_rot_points, -center, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    original_points = scale_points(inv_rot_centering_points, voxel_size, reverse=True)
    return original_points


class AxisTransformer:
    def __init__(self, axis=((1, 0, 0), (0, 1, 0), (0, 0, 1)), center=(0, 0, 0), voxel_size=(1, 1, 1)):
        self.axis = np.array(axis)
        self.center = np.array(center)
        self.voxel_size = np.array(voxel_size)

    def transform_coord(self, points_coo, voxel_size=None):
        voxel_size = self.voxel_size if voxel_size is None else voxel_size
        return transform_coord(points_coo, self.axis, self.center, voxel_size)

    def inv_transform_coord(self, points_coo, voxel_size=None):
        voxel_size = self.voxel_size if voxel_size is None else voxel_size
        return inv_transform_coord(points_coo, self.axis, self.center, voxel_size)

    def transform_napari_vector(self, vector, voxel_size=None):
        voxel_size = self.voxel_size if voxel_size is None else voxel_size
        origin_points = self.transform_coord(vector[:, 0], voxel_size=voxel_size)
        new_vector = transform_coord(vector[:, 1], self.axis, (0, 0, 0), voxel_size)
        return np.stack([origin_points, new_vector], axis=1)

    def inv_transform_napari_vector(self, vector, voxel_size=None):
        voxel_size = self.voxel_size if voxel_size is None else voxel_size
        origin_points = self.inv_transform_coord(vector[:, 0], voxel_size=voxel_size)
        new_vector = inv_transform_coord(vector[:, 1], self.axis, (0, 0, 0), voxel_size)
        return np.stack([origin_points, new_vector], axis=1)

    def axis_vectors(self):
        axis_vectors = np.zeros((3, 2, 3))
        axis_vectors[:, 1, :] = self.axis
        return axis_vectors

    def main_axis_grs(self):
        return self.axis[0]

    def secondary_axis_grs(self):
        return self.axis[1]

    def scale_volumes(self, volumes):
        return np.prod(self.voxel_size) * volumes


def find_label_com(cell_labels, cell_com, label=(1,)):
    _label_com = []
    for _com, _label in zip(cell_com, cell_labels):
        if _label in label:
            _label_com.append(_com)
        
    if len(_label_com) > 0:
        return np.mean(_label_com, axis=0)
    else:
        return None, None, None
    

def find_axis_l123(cell_labels, cell_com, l123_set=(1, 2, 3)):
    labels_com = []
    for key in l123_set:
        _com = find_label_com(cell_labels, cell_com, (key,))
        if _com[0] is not None:
            labels_com.append(_com)
    
    line = Line.best_fit(labels_com)
    # main axis aligned with labels com
    main_axis = -line.vector.unit()
    
    plane = Plane(point=labels_com[-1], normal=main_axis)
    proj_pivot_point1 = plane.project_point(labels_com[-1])
    proj_pivot_point2 = plane.project_point([0, 0, 0])
    line = Line.from_points(proj_pivot_point1, proj_pivot_point2)
    second_axis = line.vector.unit()
    
    # third axis 
    third_axis = main_axis.cross(second_axis)
    return main_axis, second_axis, third_axis


def find_axis_late(cell_labels, cell_com, l_set=(2, 3, 4, 5, 8, 14), funiculum=7):
    labels_com = []
    for key in l_set:
        _com = find_label_com(cell_labels, cell_com, (key,))
        if _com[0] is not None:
            labels_com.append(_com)
            
    line = Line.best_fit(labels_com)
    # main axis aligned with labels com
    main_axis = - line.vector
    
    # secondary axis
    if isinstance(funiculum, int):
        pivot_point = Point(find_label_com(cell_labels, cell_com, (funiculum,)))

    elif isinstance(funiculum, (list, tuple)):
        pivot_point = Point(funiculum)

    else:
        return None, None, None

    proj_pivot_point = line.project_point(pivot_point)
    second_axis = (Vector(proj_pivot_point) - Vector(pivot_point)).unit()
    
    # third axis 
    third_axis = main_axis.cross(second_axis)
    return main_axis, second_axis, third_axis 
