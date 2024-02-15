import numbers
import numpy as np
from collections import defaultdict

def __difference(a, b):
    return abs(a - b)

def __norm(p):
    return lambda a, b: np.linalg.norm(np.atleast_1d(a) - np.atleast_1d(b), p)

def limited_dynamical_programming(x, y, radius=1, distances=None):
    x, y, distances = __preprocessing_inputs(x, y, distances)
    return __limited_dynamical_programming(x, y, radius, distances)

#  me
def __limited_dynamical_programming(x, y, radius, distances):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dynamical_programming(x, y, distances=distances)

    x_shrinked, y_shrinked = __reduce_by_half(x), __reduce_by_half(y)
    distance, path = __limited_dynamical_programming(x_shrinked, y_shrinked, radius=radius, distances=distances)
    window = __expand_window(path, len(x), len(y), radius)
    return __dynamical_programming(x, y, window, distances=distances)


def __preprocessing_inputs(x, y, distances):
    x, y = np.asanyarray(x, dtype='float'), np.asanyarray(y, dtype='float')
    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(distances, numbers.Number) and distances <= 0:
        raise ValueError('distances cannot be a negative integer')
    if distances is None:
        distances=__difference if x.ndim == 1 else __norm(p=1)
    elif isinstance(distances, numbers.Number):
        distances = __norm(p=distances)
    return x, y, distances


def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius+1)
                     for b in range(-radius, radius+1)):
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1), (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = list()
    start_j = 0
    for i in range(0, len_x):
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









