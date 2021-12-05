import numpy as np
# from numba import jit
#
# @jit
def dist_xyz(point_a, point_b):
    d_point = np.array(point_a - point_b)
    distance = np.sqrt((d_point ** 2).sum())
    return distance
