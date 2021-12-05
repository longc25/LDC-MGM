import numpy as np

def get_xyz(data):
    """
    :param data: 3D data
    :return: 3D data coordinates
    第1,2,3维数字依次递增
    """
    nim = data.ndim
    if nim == 3:
        size_x, size_y, size_z = data.shape
        x_arange = np.arange(1, size_x+1)
        y_arange = np.arange(1, size_y+1)
        z_arange = np.arange(1, size_z+1)
        [xx, yy, zz] = np.meshgrid(x_arange, y_arange, z_arange, indexing='ij')
        xyz = np.column_stack([zz.flatten(), yy.flatten(), xx.flatten()])

    else:
        """
            :param data: 2D data
            :return: 2D data coordinates
            第1,2维数字依次递增
            """
        size_x, size_y = data.shape
        x_arange = np.arange(1, size_x + 1)
        y_arange = np.arange(1, size_y + 1)
        [xx, yy] = np.meshgrid(x_arange, y_arange, indexing='ij')
        xyz = np.column_stack([yy.flatten(), xx.flatten()])
    return xyz


if __name__ == '__main__':
    data = np.zeros([120,100,80])
    xx = get_xyz(data)
    print(xx[[1,200,500,1000,-1],:])

    xx_2d = get_xyz(np.zeros([110, 70]))
    print(xx_2d[[1,6,-1],:])