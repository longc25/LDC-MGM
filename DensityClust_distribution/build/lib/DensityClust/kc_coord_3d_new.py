import numpy as np
import time


def setdiff_nd(a1, a2):
    """
    python 使用numpy求二维数组的差集
    :param a1:
    :param a2:
    :return:
    """
    # a1 = index_value
    # a2 = np.array([point_ii_xy])
    a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
    a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])

    a3 = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1])
    return a3


def kc_coord_3d(delta_ii_xy, xm, ym, zm, r):
    """
    :param delta_ii_xy: 当前点坐标(x,y,z)
    :param xm: size_x
    :param ym: size_y
    :param zm: size_z
    :param r: 2 * r + 1
    :return:
    返回delta_ii_xy点r邻域的点坐标
    """
    it = delta_ii_xy[0]
    jt = delta_ii_xy[1]
    kt = delta_ii_xy[2]

    xyz_min = np.array([[1, it - r], [1, jt - r], [1, kt - r]])
    xyz_min = xyz_min.max(axis=1)

    xyz_max = np.array([[xm, it + r], [ym, jt + r], [zm, kt + r]])
    xyz_max = xyz_max.min(axis=1)

    x_arange = np.arange(xyz_min[0], xyz_max[0] + 1)
    y_arange = np.arange(xyz_min[1], xyz_max[1] + 1)
    v_arange = np.arange(xyz_min[2], xyz_max[2] + 1)

    [p_k, p_i, p_j] = np.meshgrid(x_arange, y_arange, v_arange, indexing='ij')
    Index_value = np.column_stack([p_k.flatten(), p_i.flatten(), p_j.flatten()])


    Index_value = setdiff_nd(Index_value, np.array([delta_ii_xy]))

    # t0 = time.time()
    ordrho_jj = np.matmul(Index_value - 1, np.array([[1], [xm], [ym * xm]]))
    ordrho_jj.reshape([1, ordrho_jj.shape[0]])
    # t1 = time.time()
    # print((t1-t0) * 1000000000)
    # t0 = time.time()
    # ordrho_jj = np.array([((item[2] - 1) * ym * xm + (item[1] - 1) * xm + item[0] - 1) for item in Index_value])
    # t1 = time.time()
    # print((t1 - t0) * 1000000000)
    return ordrho_jj[:, 0], Index_value


def kc_coord_2d(delta_ii_xy, xm, ym, r):
    """

    :param delta_ii_xy: 当前点坐标(x,y)
    :param xm: size_x
    :param ym: size_y
    :param r: 2 * r + 1
    :return:
    返回delta_ii_xy点r邻域的点坐标
    """
    it = delta_ii_xy[0]
    jt = delta_ii_xy[1]


    p_i, p_j = np.mgrid[max(1, it - r): min(xm, it + r) + 1, max(1, jt - r): min(ym, jt + r) + 1]

    index_value = np.zeros([p_i.size, 2], np.int)
    index_value[:, 0] = p_i.flatten()
    index_value[:, 1] = p_j.ravel()

    index_value = np.array([item for item in index_value if not all(item == delta_ii_xy)])

    return index_value


if __name__ == '__main__':
    xm, ym, zm = 100, 80, 120
    r = 3
    delta_ii_xy = [43, 22, 109]
    t0 = time.time()
    index, index_value = kc_coord_3d(delta_ii_xy, xm, ym, zm, r)
    t1 = time.time()
    print((t1-t0) * 10000000)

    # size_x, size_y, size_z = 100, 80, 120
    # x_arange = np.arange(0, size_x)
    # y_arange = np.arange(0, size_y)
    # z_arange = np.arange(0, size_z)
    # [xx, yy, zz] = np.meshgrid(x_arange, y_arange, z_arange, indexing='ij')
    # xyz = np.column_stack([zz.flat, yy.flat, xx.flat])
    #
    # # size_x, size_y, size_z = data.shape
    # # p_i, p_j, p_k = np.mgrid[0: size_x, 0: size_y, 0: size_z]
    # p_i, p_j, p_k = np.mgrid[1: size_x + 1, 1: size_y + 1, 1: size_z + 1]
    # xx1 = np.zeros([p_i.size, 3], dtype=int)
    # xx1[:, 0] = p_k.flatten()
    # xx1[:, 1] = p_j.flatten()
    # xx1[:, 2] = p_i.flatten()
