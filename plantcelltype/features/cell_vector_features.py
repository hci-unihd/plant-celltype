import numpy as np
from skspatial.objects import Vector

from numba import njit, prange
from plantcelltype.features.rag import build_nx_graph
from plantcelltype.features.utils import check_valid_idx, fibonacci_sphere
from plantcelltype.utils.utils import create_edge_mapping, create_cell_mapping, cantor_sym_pair


def compute_local_reference_axis1(cell_ids, edges_ids, cell_hops_to_bg, cell_com, edges_plane_vectors):
    """ Compute the surface axes """
    nx_graph = build_nx_graph(cell_ids, edges_ids)
    cell_com_mapping = create_cell_mapping(cell_ids, cell_com)
    cell_hops_to_bg_mapping = create_cell_mapping(cell_ids, cell_hops_to_bg)
    edges_pv_mapping = create_edge_mapping(edges_ids, edges_plane_vectors)
    lr_axis_1 = np.zeros((cell_ids.shape[0], 3))
    for i, c_idx in enumerate(cell_ids):
        list_nx = list(nx_graph.neighbors(c_idx))
        c_idx_hops = cell_hops_to_bg_mapping[c_idx]
        c_com = cell_com_mapping[c_idx]

        if c_idx_hops == 1:
            local_vectors = Vector(edges_pv_mapping[cantor_sym_pair(0, c_idx)][1])

        else:
            local_vectors = Vector([0.0, 0.0, 0.0])
            for cn_idx in list_nx:
                cn_idx_hops = cell_hops_to_bg_mapping[cn_idx]
                if cn_idx_hops < c_idx_hops:
                    edges_pv = edges_pv_mapping[cantor_sym_pair(cn_idx, c_idx)]
                    vector = Vector(edges_pv[1])
                    test_vector = Vector.from_points(c_com, edges_pv[0])
                    if vector.dot(test_vector) < 0:
                        vector *= -1

                    local_vectors += vector

        local_vectors = local_vectors.unit()
        lr_axis_1[i] = local_vectors

    return lr_axis_1


def get_query_vector(cell_com_mapping, a_nx, com_n):
    com_nx = cell_com_mapping[a_nx]
    cv = Vector.from_points(com_n, com_nx)
    return cv.unit()


def local_vectors_alignment(nx_graph, vectors_mapping, iteration=10):
    for _ in range(iteration):
        for n, current_vector in vectors_mapping.items():
            list_nx = list(nx_graph.neighbors(n))

            energy = 0.
            for i, v_n in enumerate(list_nx):
                energy += current_vector.dot(vectors_mapping[v_n])

            if energy < 0.:
                vectors_mapping[n] = - current_vector

    return vectors_mapping


def local_vectors_averaging(nx_graph, vectors_mapping, iteration=10, alpha=0.1):
    for _ in range(iteration):
        for n, current_vector in vectors_mapping.items():
            list_nx = list(nx_graph.neighbors(n))

            mean_vector = Vector(np.zeros(3))
            for i, v_n in enumerate(list_nx):
                mean_vector += vectors_mapping[v_n]

            mean_vector = mean_vector / len(list_nx)
            mean_vector = vectors_mapping[n] + alpha * mean_vector
            vectors_mapping[n] = mean_vector.unit()

    return vectors_mapping


def compute_local_reference_axis2(cell_ids,
                                  edges_ids,
                                  cell_com,
                                  cell_hops_to_bg,
                                  global_axis=(1, 0, 0)):
    """ Compute the growth axes """
    global_axis = Vector(global_axis)
    nx_graph = build_nx_graph(cell_ids, edges_ids)
    cell_com_mapping = create_cell_mapping(cell_ids, cell_com)
    cell_hops_to_bg_mapping = create_cell_mapping(cell_ids, cell_hops_to_bg)

    # lr_axis_2_mapping = np.zeros((cell_ids.shape[0], 3))  # {}
    lr_axis_2 = np.zeros((cell_ids.shape[0], 3))  # {}
    lr_axis_2_angle = np.zeros((cell_ids.shape[0]))
    for idx, n in enumerate(cell_ids):
        # defaults
        min_dot, lr_vector = 1., global_axis

        hops_n = cell_hops_to_bg_mapping[n]
        list_nx = [nx for nx in nx_graph.neighbors(n) if hops_n == cell_hops_to_bg_mapping[nx]]
        if len(list_nx) == 0:
            list_nx = list(nx_graph.neighbors(n))

        com_n = cell_com_mapping[n]
        for i, a_n1 in enumerate(list_nx):
            # test 1/2 query cell 1
            cv1 = get_query_vector(cell_com_mapping, a_n1, com_n)
            for a_n2 in list_nx[i + 1:]:
                # test 1/2 query cell 2
                cv2 = get_query_vector(cell_com_mapping, a_n2, com_n)

                total_similarity = cv1.dot(cv2)

                if total_similarity < min_dot:
                    min_dot = total_similarity
                    # find orientation more in line with lr vector
                    fv1_g_sym = cv1.dot(global_axis)
                    fv2_g_sym = cv2.dot(global_axis)
                    lr_vector = cv1 if fv1_g_sym > fv2_g_sym else cv2

        # lr_axis_2_mapping[n] = lr_vector
        lr_axis_2[idx] = lr_vector
        lr_axis_2_angle[idx] = min_dot

    # removed for consistency
    # lr_axis_2_mapping = local_vectors_alignment(nx_graph, lr_axis_2_mapping, iteration=10)
    # lr_axis_2_mapping = local_vectors_averaging(nx_graph, lr_axis_2_mapping)
    # lr_axis_2 = np.array([lr_axis_2_mapping[idx] for idx in cell_ids])
    return lr_axis_2, lr_axis_2_angle


def compute_local_reference_axis3(cell_axis1, cell_axis2):
    cell_axis3 = np.zeros_like(cell_axis1)
    for i, (ax1, ax2) in enumerate(zip(cell_axis1, cell_axis2)):
        ax3 = Vector(ax1).cross(ax2)
        cell_axis3[i] = ax3
    return cell_axis3


@njit
def _l2_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


@njit
def _test_angles(valid_samples, cell_com_i, vector):
    # 2x faster than pure python
    min_point, max_point = np.zeros(3, dtype='float32'), np.zeros(3, dtype='float32')
    min_value, max_value = 0, 0
    for sample in valid_samples:
        test_vector = sample - cell_com_i
        test_vector /= np.sqrt(np.sum(test_vector ** 2))
        value = np.dot(vector, test_vector)

        if value < min_value:
            min_value, min_point = value, sample

        elif value > max_value:
            max_value, max_point = value, sample

    return min_point, max_point


@njit(parallel=True)
def compute_length_along_axis(cell_com, cell_samples, axis, origin=(0, 0, 0)):
    n_samples = cell_com.shape[0]
    lengths = np.zeros(n_samples)
    for i in prange(n_samples):
        # create axis vector
        cell_com_i = cell_com[i]
        sample = cell_samples[i]

        # remove empty samples
        valid_samples, _ = check_valid_idx(sample, origin)

        # find min and max point along the axis
        _, max_point = _test_angles(valid_samples, cell_com_i, axis)
        # evaluate length
        lengths[i] = _l2_distance(cell_com_i, max_point)

    return lengths


def compute_proj_length_on_sphere(cell_com, cell_samples, n_samples=64, origin=(0, 0, 0)):
    sample_on_unit_sphere = np.array(fibonacci_sphere(n_samples)).astype('float32')
    origin = np.array(origin).astype('float32')

    length = np.zeros((cell_com.shape[0], sample_on_unit_sphere.shape[0]))
    for i, axis in enumerate(sample_on_unit_sphere):
        length[:, i] = compute_length_along_axis(cell_com, cell_samples, axis, origin=origin)
    return length


def compute_sym_length_along_cell_axis(cell_axis, cell_com, cell_samples, origin=(0, 0, 0)):
    lengths = np.zeros(cell_com.shape[0])
    for i, (com_i, sample_i, axis_i) in enumerate(zip(cell_com, cell_samples, cell_axis)):
        # remove empty samples
        valid_samples, _ = check_valid_idx(sample_i, origin)

        # find min and max point along the axis
        min_point, max_point = _test_angles(valid_samples, com_i, axis_i)
        # evaluate length
        lengths[i] = _l2_distance(min_point, max_point)
    return lengths


def get_vectors_orientation_mapping(vectors_array):
    orientation_vectors_array = np.zeros((vectors_array.shape[0], 6))
    orientation_vectors_array[:, :3] = vectors_array ** 2
    # additional
    orientation_vectors_array[:, 3] = vectors_array[:, 0] * vectors_array[:, 1]
    orientation_vectors_array[:, 4] = vectors_array[:, 1] * vectors_array[:, 2]
    orientation_vectors_array[:, 5] = vectors_array[:, 2] * vectors_array[:, 0]
    return orientation_vectors_array.astype('float32')
