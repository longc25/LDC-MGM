from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def gaussian_3d_rotate(A, x0, y0, v0, s1, s2, s3, theta):
    x0 = float(x0)
    y0 = float(y0)
    v0 = float(v0)
    theta = theta / 180 * np.pi
    f = lambda x: A * np.exp(-(((x[:, 0] - x0) ** 2) * (np.cos(theta) ** 2 / (2 * s1 ** 2) + np.sin(theta) ** 2 / (2 * s2 ** 2)) +
                 ((x[:, 1] - y0) ** 2) * (np.sin(theta) ** 2 / (2 * s1 ** 2) + np.cos(theta) ** 2 / (2 * s2 ** 2)) +
                 (x[:, 0] - x0) * (x[:, 1] - y0) * (
                             2 * (-np.sin(2 * theta) / (4 * s1 ** 2) + np.sin(2 * theta) / (4 * s2 ** 2))) +
                 ((x[:, 2] - v0) ** 2) / (2 * s3 ** 2)))

    g = f
    return g

def gaussian_3d_rotate_multi(A, x0, s1, y0, s2, theta, v0, s3):
    # x0 = float(x0)
    # y0 = float(y0)
    # v0 = float(v0)
    # theta = theta / 180 * np.pi
    # f_str += '+A[%s] * np.exp(-(((x[:, 0] - x0[%s]) ** 2) * (np.cos(theta[%s]) ** 2 / (2 * s1[%s] ** 2) + \
    #         np.sin(theta[%s]) ** 2 / (2 * s2[%s] ** 2)) +\
    #         ((x[:, 1] - y0[%s]) ** 2) * (np.sin(theta[%s]) ** 2 / (2 * s1[%s] ** 2) + np.cos(theta[%s]) ** 2 / (2 * s2[%s] ** 2)) +\
    #         (x[:, 0] - x0[%s]) * (x[:, 1] - y0[%s]) * (\
    #         2 * (-np.sin(2 * theta[%s]) / (4 * s1[%s] ** 2) +\
    #          np.sin(2 * theta[%s]) / (4 * s2[%s] ** 2))) + \
    #          ((x[:, 2] - v0[%s]) ** 2) / (2 * s3[%s] ** 2)))' % (
    # i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i)
    #
    # f_str += '+A[%s, 0] * np.exp(-(((x[:, 0] - A[%s, 1]) ** 2) * (np.cos(A[%s,7]) ** 2 / (2 * A[%s, 4] ** 2) + \
    #         np.sin(A[%s,7]) ** 2 / (2 * A[%s,5] ** 2)) +\
    #         ((x[:, 1] - A[%s,2]) ** 2) * (np.sin(A[%s,7]) ** 2 / (2 * A[%s,4] ** 2) + np.cos(A[%s,7]) ** 2 / (2 * A[%s,5] ** 2)) +\
    #         (x[:, 0] - A[%s,1]) * (x[:, 1] - A[%s,2]) * (\
    #         2 * (-np.sin(2 * A[%s,7]) / (4 * A[%s,4] ** 2) +\
    #          np.sin(2 * A[%s,7]) / (4 * A[%s,5] ** 2))) + \
    #          ((x[:, 2] - A[%s,3]) ** 2) / (2 * A[%s,6] ** 2)))' % (
    # i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i)
    f_str = ''
    # print(A.shape)
    # n_dim = A.ndim
    # # print('N_dim is %d' %n_dim)
    # if n_dim == 1:
    #     # print(A.shape[0])
    #     num = int(A.shape[0] / 8)
    #     A = np.reshape(A, [num,8])
    for i in range(len(A)):
        # print(i)
        f_str += '+A[%s] * np.exp(-(((x[:, 0] - x0[%s]) ** 2) * (np.cos(theta[%s]) ** 2 / (2 * s1[%s] ** 2) + \
                    np.sin(theta[%s]) ** 2 / (2 * s2[%s] ** 2)) +\
                    ((x[:, 1] - y0[%s]) ** 2) * (np.sin(theta[%s]) ** 2 / (2 * s1[%s] ** 2) + np.cos(theta[%s]) ** 2 / (2 * s2[%s] ** 2)) +\
                    (x[:, 0] - x0[%s]) * (x[:, 1] - y0[%s]) * (\
                    2 * (-np.sin(2 * theta[%s]) / (4 * s1[%s] ** 2) +\
                     np.sin(2 * theta[%s]) / (4 * s2[%s] ** 2))) + \
                     ((x[:, 2] - v0[%s]) ** 2) / (2 * s3[%s] ** 2)))' % (i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i)
    f_str = f_str[1:]
    func_exp = eval(f_str)
    f = lambda x: func_exp

    return f

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def moments_3d(data,data1):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data1.sum()
    X, Y, Z = np.indices(data1.shape)
    x = (X*data1).sum()/total
    y = (Y*data1).sum()/total
    v = (Z * data1).sum() / total
    col = data1[:, int(y),:]
    col = col.flatten()
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :,:]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())

    v_rr = data[:,:,int(v)]
    width_v = np.sqrt(np.abs((np.arange(v_rr.size) - v) ** 2 * v_rr).sum() / row.sum())
    height = data.max()
    angle = 0


    return height, x, y, v, width_x, width_y, width_v, angle

def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)

    errorfunction = lambda p: np.ravel((gaussian(*p)(*np.indices(data.shape)) -
                                 data) * (data**4 / (data**4).sum()))
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fitgaussian_3d(X, data, param):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    # params = moments_3d(data)
    params = param
    # Xin, Yin, Vin = np.mgrid[0:31, 0:31, 0:31]
    # X = np.vstack([Xin.flatten(), Yin.flatten(), Vin.flatten()]).T
    power = 2
    errorfunction = lambda p: np.ravel((gaussian_3d_rotate(*p)(X) -
                                        data) * (data ** power / (data ** power).sum()))
    p, success = optimize.leastsq(errorfunction, params)
    return p


def fitgaussian_3d_multi(X, data, A):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    # params = moments_3d(data)
    params = A
    # print("A shape is %d*%d" % A.shape)
    # Xin, Yin, Vin = np.mgrid[0:31, 0:31, 0:31]
    # X = np.vstack([Xin.flatten(), Yin.flatten(), Vin.flatten()]).T
    power = 4
    errorfunction = lambda p: np.ravel((gaussian_3d_rotate_multi(p)(X) -
                                        data) * (data ** power / (data ** power).sum()))
    p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction,x0=params,full_output=1)
    return p, cov, success

def fit_gauss_2d():
    Xin, Yin = np.mgrid[0:31, 0:31]
    data = gaussian(3, 15, 15, 5, 5)(Xin, Yin) + 0.3 * np.random.random(Xin.shape)

    plt.matshow(data, cmap=plt.cm.gist_earth_r)

    params = fitgaussian(data)
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params

    plt.text(0.95, 0.05, """
        x : %.1f
        y : %.1f
        width_x : %.1f
        width_y : %.1f""" % (x, y, width_x, width_y),
             fontsize=16, horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes)


if __name__ == '__main__':
    Xin, Yin, Vin = np.mgrid[0:31, 0:31, 0:31]
    X = np.vstack([Xin.flatten(), Yin.flatten(), Vin.flatten()]).T
    param = np.array([6, 15, 15, 15, 3, 5, 7, 15])
    data = gaussian_3d_rotate(*param)(X)
    ind = np.where(data > 0.5)[0]
    X1 = X[ind, ...]
    Y1 = data[ind, ...]

    noise = 1 * np.random.randn(Y1.shape[0])
    params = fitgaussian_3d(X1, Y1 + noise, param+2)
    data1 = (data + 0.1 * np.random.randn(data.shape[0])).reshape(Xin.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for i, ax_item in enumerate([ax1, ax2, ax3]):
        ax_item.imshow(data1.sum(axis=i))
    plt.show()

    print('simulated parameters for 3d gauss:')
    print(param)

    print('fitting parameters for 3d gauss:')
    print(params)

    print('*'*50)
    A, x0, s1, y0, s2, theta, v0, s3 = (6,0,0), (10,20,15), (10,20,15), (10, 20,15), (3, 3,3), (3, 3,3), (3, 3,3), (15/180*np.pi,75/180*np.pi,45/180*np.pi)
    # A = np.array([(6,6,5), (10,20,15), (10,20,15), (10, 20,15), (3, 3,4), (5, 5,4), (7, 7,6), (15/180*np.pi,75/180*np.pi,45/180*np.pi)]).T
    data_2 = f(X)

    data1 = (data_2).reshape(Xin.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for i, ax_item in enumerate([ax1, ax2, ax3]):
        ax_item.imshow(data1.sum(axis=i))
    plt.show()

    ind = np.where(data_2 > 0.01)[0]
    X2 = X[ind, ...]
    Y2 = data_2[ind, ...]

    noise = np.random.randn(Y2.shape[0])
    params, cov, success = fitgaussian_3d_multi(X2, Y2 + noise, A+0.3)
    params = params.reshape(A.shape)
    params[:,7] = params[:,7] / np.pi * 180

    A[:, 7] = A[:, 7] / np.pi * 180
    print('simulated parameters for multi 3d gauss:\n')
    print(A)

    print('fitting parameters for multi 3d gauss:\n')
    print(params)


    # fit = gaussian(*params)
    #
    # plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    # ax = plt.gca()
    # (height, x, y, width_x, width_y) = params
    #
    # plt.text(0.95, 0.05, """
    #     x : %.1f
    #     y : %.1f
    #     width_x : %.1f
    #     width_y : %.1f""" % (x, y, width_x, width_y),
    #          fontsize=16, horizontalalignment='right',
    #          verticalalignment='bottom', transform=ax.transAxes)
    import sympy as sp
    sp.lambdify




