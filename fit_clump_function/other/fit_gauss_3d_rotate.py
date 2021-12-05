import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from LDC_version.DensityClust.make_plot import make_plot
from LDC_version.DensityClust.get_xx import get_xyz
# from sklearn.linear_model import RANSACRegressor


def gauss_3d_rotate(x,A,x0,s1,y0,s2,theta,v0,s3):
    """
    三维高斯分布，在x-y平面上存在旋转
    :param x: [x,y,v] M*3的数组
    :param A: peak
    :param x0: Cen1
    :param s1: size1
    :param y0: Cen2
    :param s2: size2
    :param theta: 旋转角
    :param v0: Cen3
    :param s3: size3
    :return:
    M*1的数组
    """
    return A * np.exp( -(((x[:, 0]-x0)**2) * (np.cos(theta)**2 / (2*s1**2) + np.sin(theta)**2 / (2*s2**2)) +
                         ((x[:, 1]-y0)**2) * (np.sin(theta)**2 / (2*s1**2) + np.cos(theta)**2 / (2*s2**2)) +
                         (x[:, 0]-x0) * (x[:, 1]-y0) * (2*(-np.sin(2*theta) / (4*s1**2) + np.sin(2*theta) / (4*s2**2))) +
                         ((x[:, 2]-v0)**2) / (2*s3**2)))

def gaussian_3d_rotate_multi(A):
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
    f_str = ''
    # print(A.shape)
    n_dim = A.ndim
    # print('N_dim is %d' %n_dim)
    if n_dim == 1:
        # print(A.shape[0])
        num = int(A.shape[0] / 8)
        A = np.reshape(A, [num,8])
    for i in range(A.shape[0]):
        # print(i)
        f_str += '+A[%s, 0] * np.exp(-(((x[:, 0] - A[%s, 1]) ** 2) * (np.cos(A[%s,7]) ** 2 / (2 * A[%s, 4] ** 2) + \
        np.sin(A[%s,7]) ** 2 / (2 * A[%s,5] ** 2)) +\
        ((x[:, 1] - A[%s,2]) ** 2) * (np.sin(A[%s,7]) ** 2 / (2 * A[%s,4] ** 2) + np.cos(A[%s,7]) ** 2 / (2 * A[%s,5] ** 2)) +\
        (x[:, 0] - A[%s,1]) * (x[:, 1] - A[%s,2]) * (\
        2 * (-np.sin(2 * A[%s,7]) / (4 * A[%s,4] ** 2) +\
         np.sin(2 * A[%s,7]) / (4 * A[%s,5] ** 2))) + \
         ((x[:, 2] - A[%s,3]) ** 2) / (2 * A[%s,6] ** 2)))' %(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)
    f_str = f_str[1:]
    f = lambda x: eval(f_str)

    return f

if __name__ == '__main__':

    data = np.zeros([100,100,100])
    xx = get_xyz(data)

    A,x0,s1,y0,s2,theta,v0,s3 = 10,50.2,5,50.4,10,30/180*np.pi,50.8,5

    gauss_data = gauss_3d_rotate(xx,A,x0,s1,y0,s2,theta,v0,s3)
    gauss_noise = np.random.random(gauss_data.shape)
    gauss_data_3d = np.reshape(gauss_data,data.shape)

    make_plot(None, gauss_data_3d + np.reshape(gauss_noise,data.shape))
    print(np.where(gauss_data_3d==gauss_data_3d.max()))
    print(gauss_data_3d.max())
    data = fits.getdata(r'../test_data/test.fits')
    xx = get_xyz(data)
    y = data.transpose(2,1,0).flatten()
    idx = np.where(y>0.01)[0]
    xx1 = xx[idx,:]
    y1 = y[idx]

    popt, pcov = curve_fit(gauss_3d_rotate, xx1, y1, p0=[A+1,x0+5,s1+4,y0+5,s2+4,theta+0.4,v0+5,s3+4])
    print(popt)

    popt, pcov = curve_fit(gauss_3d_rotate, xx, y)
    print(popt)