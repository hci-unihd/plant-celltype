import numpy as np
from skspatial.objects import Plane, Line, Point, Vector


def find_label_com(com, labels, label):
    _label_com = []
    for _com, _label in zip(com, labels):
        if _label in label:
            _label_com.append(_com)
        
    if len(_label_com) > 0:
        return np.mean(_label_com, axis=0)
    else:
        return None, None, None
    

def find_axis_l123(com, labels, l_set=(1, 2, 3)):
    labels_com = []
    for key in l_set:
        _com = find_label_com(com, labels, (key, ))
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


def find_axis_late(com, labels, l_set=(2, 3, 4, 5, 8, 14), pivot=7):
    labels_com = []
    for key in l_set:
        _com = find_label_com(com, labels, (key, ))
        if _com[0] is not None:
            labels_com.append(_com)
            
    line = Line.best_fit(labels_com)
    # main axis aligned with labels com
    main_axis = - line.vector
    
    # secondary axis
    if isinstance(pivot, int):
        print(find_label_com(com, labels, (pivot, )))
        pivot_point = Point(find_label_com(com, labels, (pivot, )))

    elif isinstance(pivot, (list, tuple)):
        pivot_point = Point(pivot)

    else:
        return None, None, None

    proj_pivot_point = line.project_point(pivot_point)
    second_axis = (Vector(proj_pivot_point) - Vector(pivot_point)).unit()
    
    # third axis 
    third_axis = main_axis.cross(second_axis)
    return main_axis, second_axis, third_axis 
