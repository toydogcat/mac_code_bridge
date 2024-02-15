import numbers
import numpy as np
from collections import defaultdict
import tensorflow as tf


def __difference(a, b):
    return abs(a - b)

def __norm(p):
    return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)

def limited_dynamical_programming(x, y, radius=1, distances=None):
    x, y, distances = __preprocessing_inputs(x, y, distances)
    return __limited_dynamical_programming(x, y, radius, distances)

def __limited_dynamical_programming(x, y, radius, distances):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dynamical_programming(x, y, distances=distances)

    x_shrinked, y_shrinked = __reduce_by_half(x), __reduce_by_half(y)
    distance, path = __limited_dynamical_programming(x_shrinked, y_shrinked, radius=radius, distances=distances)
    window = __expand_window(path, len(x), len(y), radius)
    return __dynamical_programming(x, y, window, distances=distances)

def __preprocessing_inputs(x, y, distances):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != y.ndim or (x.ndim == 2 and x.shape[1] != y.shape[1]):
        raise ValueError('x and y must have the same number of dimensions and matching second dimension')

    if isinstance(distances, numbers.Number) and distances <= 0:
        raise ValueError('distances cannot be a negative number')

    if distances is None:
        distances = __difference if x.ndim == 1 else __norm(p=1)
    elif isinstance(distances, numbers.Number):
        distances = __norm(p=distances)

    return x, y, distances


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    window_ = set()

    for i, j in path:
        for a in range(i - radius, i + radius + 1):
            for b in range(j - radius, j + radius + 1):
                path_.add((a, b))

    for i, j in path_:
        for a in (i * 2, i * 2, i * 2 + 1, i * 2 + 1):
            for b in (j * 2, j * 2 + 1, j * 2, j * 2 + 1):
                window_.add((a, b))

    window = []
    start_j = 0
    for i in range(len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window


def dynamical_programming(x, y, distances=None):
    x, y, distances = __preprocessing_inputs(x, y, distances)
    return __dynamical_programming(x, y, None, distances)


def __dynamical_programming(x, y, window, distances):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)
    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    for i, j in window:
        dt = distances(x[i-1], y[j-1])
        D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1), (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
    path = list()
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[len_x, len_y][0], path)


def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]


# ------------------------ distance part ------------------------


def distance_from_list(keypoints_c, keypoints_u, index_list = ((5,7), (7,9))):
    dist = distance_from_idl(keypoints_c, keypoints_u, index_list)
    return dist 


def vector_from_index(keypoints, index_pair):
    keypoint_1 = keypoints[index_pair[0]]
    keypoint_2 = keypoints[index_pair[1]]
    
    keypoint_1 = keypoint_1.numpy() if tf.is_tensor(keypoint_1) else keypoint_1
    keypoint_2 = keypoint_2.numpy() if tf.is_tensor(keypoint_2) else keypoint_2

    y_1, x_1 = keypoint_1[0], keypoint_1[1]
    y_2, x_2 = keypoint_2[0], keypoint_2[1]
    vec = (x_2 - x_1, y_2 - y_1)
    return vec


def distance_from_idl(keypoints_c, keypoints_u, index_list):
    delta = 0.0001
    vec_c_12 = vector_from_index(keypoints_c, index_list[0])
    vec_c_23 = vector_from_index(keypoints_c, index_list[1])
    vec_u_12 = vector_from_index(keypoints_u, index_list[0])
    vec_u_23 = vector_from_index(keypoints_u, index_list[1])
    
    norm_vec_c_12 = np.linalg.norm(vec_c_12)
    norm_vec_c_23 = np.linalg.norm(vec_c_23)
    norm_vec_u_12 = np.linalg.norm(vec_u_12)
    norm_vec_u_23 = np.linalg.norm(vec_u_23)

    unit_vec_c_12 = vec_c_12 / norm_vec_c_12
    unit_vec_c_23 = vec_c_23 / norm_vec_c_23
    unit_vec_u_12 = vec_u_12 / norm_vec_u_12
    unit_vec_u_23 = vec_u_23 / norm_vec_u_23

    dot_product_12 = np.dot(unit_vec_c_12, unit_vec_u_12)
    if  -delta < dot_product_12 - 1 < delta:
        angle_12 = 0
    else:
        angle_12 = np.arccos(dot_product_12)
    
    dot_product_23 = np.dot(unit_vec_c_23, unit_vec_u_23)
    if -delta < dot_product_23 - 1 < delta:
        angle_23 = 0
    else:
        angle_23 = np.arccos(dot_product_23)
    
    rate_c = norm_vec_c_12 / norm_vec_c_23
    rate_u = norm_vec_u_12 / norm_vec_u_23
    rate = abs(rate_c / rate_u - 1)

    dist = abs(angle_12) + abs(angle_23) + rate
    
    return dist


if __name__ == '__main__':
    ###############################
    #    This is the test case
    #    If you want to implement, you can check you are right
    #    這是 toby 演算法的測試用例，任何要實作移植到其他地方，
    #    請測試過自己的移植是正確的．
    ###############################
    from scipy.spatial.distance import euclidean
    x = np.array([6, 1, 2, 3, 4, 5], dtype='float')
    y = np.array([2, 3, 4, 6], dtype='float')

    # (6.0, [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3)])
    result = dynamical_programming(x, y)
    print(result)

    x = np.array([[6,6], [1,1], [2,2], [3,3], [4,4], [5,5]])
    y = np.array([[2,2], [3,3], [4,4], [6,6]])

    distance, path = limited_dynamical_programming(x, y, distances=euclidean)

    # 8.485281374238571
    print(distance)
    # [(0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3)]
    print(path)

    keypoints_person_1 = [[0.13710973, 0.51133519, 0.77324641],
                          [0.12026259, 0.52805716, 0.86441517],
                          [0.120331,   0.49015069, 0.88547444],
                          [0.13337706, 0.5489583,  0.82559729],
                          [0.13742585, 0.46478772, 0.85538161],
                          [0.23065263, 0.61108941, 0.81994689],
                          [0.23405698, 0.41281193, 0.76733166],
                          [0.36531568, 0.62057596, 0.86234498],
                          [0.36230537, 0.40286618, 0.90409291],
                          [0.25307766, 0.55684775, 0.70859182],
                          [0.24549244, 0.47579038, 0.7037288 ],
                          [0.49700621, 0.56087124, 0.84792531],
                          [0.49499273, 0.45676968, 0.86217237],
                          [0.70955807, 0.60610032, 0.81061399],
                          [0.70927268, 0.41311818, 0.91913241],
                          [0.8824262,  0.62861526, 0.81046194],
                          [0.87469727, 0.38422289, 0.81039083]]

    keypoints_person_2 = [[ 0.60167509,  0.53796202,  0.57482237],
                          [ 0.53254318,  0.62243843,  0.40343064],
                          [ 0.53601027,  0.46179509,  0.44752562],
                          [ 0.57197857,  0.73688859,  0.53840864],
                          [ 0.57405216,  0.35073823,  0.40886256],
                          [ 0.85808432,  0.88330621,  0.29534733],
                          [ 0.8643617,   0.20306197,  0.25707546],
                          [ 0.85068697,  0.98697186,  0.01982217],
                          [ 0.91831082,  0.07156318,  0.07110119],
                          [ 0.76907748,  0.82782811,  0.01394222],
                          [ 0.54298139, -0.02357928,  0.01075112],
                          [ 1.00550032,  0.93960565,  0.03560628],
                          [ 1.00582206,  0.31314996,  0.08865499],
                          [ 0.48900381,  0.97816676,  0.00590879],
                          [ 0.48737583,  0.37535575,  0.00918998],
                          [ 0.50272059,  0.68553579,  0.01025856],
                          [ 0.51905346,  0.77460819,  0.00334356]]

    index_list = ((5,7), (7,9))

    # 2.95217
    print('distance : ', distance_from_list(keypoints_person_1, keypoints_person_2, index_list))








