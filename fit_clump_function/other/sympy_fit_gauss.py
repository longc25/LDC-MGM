from sympy import sin, cos, symbols, lambdify, exp
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from astropy.io import fits


def gauss_2d(A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1):
    A0 = np.array([A0, x0, y0,s0_1,s0_2,theta_0])
    A1 = np.array([A1, x1, y1,s1_1, s1_2,theta_1])
    A = np.array([A0,A1])
    paras = []
    x = symbols('x')
    y = symbols('y')
    paras.append(x)
    paras.append(y)
    num = A.shape[0]
    express1 = ''
    for i in range(num):
        temp = ' + A[%d,0] * exp(-((x - A[%d,1]) ** 2 * (cos(A[%d,5])**2 / (2 * A[%d,3]**2) + sin(A[%d,5])**2 / (2 * A[%d,4]**2)) \
        + (y - A[%d,2])**2 * (sin(A[%d,5])**2 / (2 * A[%d,3]**2) + cos(A[%d,5])**2 / (2 * A[%d,4]**2))\
        + (sin(2*A[%d,5]) / (2 * A[%d,4] ** 2) - sin(2*A[%d,5]) / (2 * A[%d,3] ** 2)) * ((x - A[%d,1]) * (y - A[%d,2])) ))'\
               %(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)

        express1 += temp
    express = express1[2:]
    express.replace(' ','')
    express1 = 'A0 * exp(-((x - x0) ** 2 / 2 + (y - y0)**2 / 2)) + A1 * exp(-((x - x1) ** 2 / 2 + (y - y1)**2 / 2))'
    g = eval(express)  # <class 'sympy.core.mul.Mul'>

    g1 = lambdify(paras, g, 'numpy')
    return g1

def gauss_2d_A(A):

    # A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1])
    param_num = 6
    num = A.shape[0]
    num_j = num // param_num
    paras = []
    x = symbols('x')
    y = symbols('y')
    paras.append(x)
    paras.append(y)
    express1 = ''
    for i in range(num_j):
        temp = ' + A[%d*6+0] * exp(-((x - A[%d*6+1]) ** 2 * (cos(A[%d*6+5])**2 / (2 * A[%d*6+3]**2) + sin(A[%d*6+5])**2 / (2 * A[%d*6+4]**2)) \
        + (y - A[%d*6+2])**2 * (sin(A[%d*6+5])**2 / (2 * A[%d*6+3]**2) + cos(A[%d*6+5])**2 / (2 * A[%d*6+4]**2))\
        + (sin(2*A[%d*6+5]) / (2 * A[%d*6+4] ** 2) - sin(2*A[%d*6+5]) / (2 * A[%d*6+3] ** 2)) * (x - A[%d*6+1]) * (y - A[%d*6+2]) ))'\
               %(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)

        express1 += temp
    express = express1[2:]
    express.replace(' ','')
    express1 = 'A0 * exp(-((x - x0) ** 2 / 2 + (y - y0)**2 / 2)) + A1 * exp(-((x - x1) ** 2 / 2 + (y - y1)**2 / 2))'
    g = eval(express)  # <class 'sympy.core.mul.Mul'>

    g1 = lambdify(paras, g, 'numpy')
    return g1


def gauss_3d_A(A):

    # A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3, A1, x1, y1, s1_1, s1_2,theta_1,v1, s1_3])
    param_num = 8
    num = A.shape[0]
    num_j = num // param_num
    paras = []
    x = symbols('x')
    y = symbols('y')
    v = symbols('v')
    paras.append(x)
    paras.append(y)
    paras.append(v)
    express1 = ''
    for i in range(num_j):
        temp = ' + A[%d*8+0] * exp(-((x - A[%d*8+1]) ** 2 * (cos(A[%d*8+5])**2 / (2 * A[%d*8+3]**2) + sin(A[%d*8+5])**2 / (2 * A[%d*8+4]**2)) \
        + (y - A[%d*8+2])**2 * (sin(A[%d*8+5])**2 / (2 * A[%d*8+3]**2) + cos(A[%d*8+5])**2 / (2 * A[%d*8+4]**2))\
        + (sin(2*A[%d*8+5]) / (2 * A[%d*8+4] ** 2) - sin(2*A[%d*8+5]) / (2 * A[%d*8+3] ** 2)) * (x - A[%d*8+1]) * (y - A[%d*8+2])\
         + (v - A[%d*8+6]) ** 2 / (2 * A[%d*8+7]**2) ))'\
               % (i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)

        express1 += temp
    express = express1[2:]
    express.replace(' ','')
    express1 = 'A0 * exp(-((x - x0) ** 2 / 2 + (y - y0)**2 / 2)) + A1 * exp(-((x - x1) ** 2 / 2 + (y - y1)**2 / 2))'
    g = eval(express)  # <class 'sympy.core.mul.Mul'>

    g1 = lambdify(paras, g, 'numpy')
    return g1

A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1 = 5,10,10, 2,3,0/180*np.pi, 9,20,20,2,4,45/180*np.pi
A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, A1, x1, y1, s1_1, s1_2,theta_1, 10, 15,15,2,2,0])
gauss_2d_11 = gauss_2d_A(A)

print(gauss_2d_11(20,20))
print(gauss_2d_11(10,10))

Xin, Yin = np.mgrid[0:31, 0:31]
X = np.vstack([Xin.flatten(), Yin.flatten()]).T

Y = gauss_2d_11(X[:,0],X[:,1])

data1 = Y.reshape(Xin.shape)
plt.imshow(data1)
plt.show()


def fit_gauss_2d(X,Y,params):
    power = 4
    errorfunction = lambda p: np.ravel((gauss_2d_A(p)(X[:,0],X[:,1]) - Y) * (Y ** power / (Y ** power).sum()))
    p, success = optimize.leastsq(errorfunction, x0=params)
    return p

def fit_gauss_3d(X,Y,params):
    power = 4
    weight = None  # 创建拟合的权重
    errorfunction = lambda p: np.ravel((gauss_3d_A(p)(X[:,0],X[:,1],X[:,2]) - Y) * (Y ** power / (Y ** power).sum()))
    p, success = optimize.leastsq(errorfunction, x0=params)
    return p

# Y = Y + np.random.randn(Y.shape[0])
ind = np.where(Y > 0.5)[0]
X2 = X[ind, ...]
Y2 = Y[ind, ...]

params = A - 1

p = fit_gauss_2d(X,Y,params)
print(p)

print(p[5]/np.pi * 180)
print(p[10]/np.pi * 180)

# 对三维高斯云核进行拟合

A0, x0, y0, s0_3,s0_2, theta_0, v0, s0_1 = 5,10,12, 2,4,30/180*np.pi, 13,6
A1, x1, y1, s1_3,s1_2, theta_1, v1, s1_1 = 8,18,21, 2,4,76/180*np.pi, 16,6
A = np.array([A0, x0, y0, s0_1,s0_2, theta_0, v0, s0_3,A1, x1, y1, s1_1,s1_2, theta_1, v1, s1_3])
gauss_3d_11 = gauss_3d_A(A)

Xin, Yin, Vin = np.mgrid[0:31, 0:41, 0:51]
X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T

Y = gauss_3d_11(X[:,0],X[:,1],X[:,2])
Y1 = gauss_3d_11(X[:,2],X[:,1],X[:,1])

a = Y - Y1
# data1 = Y.reshape(Xin.shape)
data1 = Y.reshape(Yin.shape)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
for i, ax_item in enumerate([ax1, ax2, ax3]):
    ax_item.imshow(data1.sum(axis=i))
plt.show()


ind = np.where(Y > 0.5)[0]
X2 = X[ind, ...]
Y2 = Y[ind, ...]

params = A - 1

p = fit_gauss_3d(X,Y,params)
print(p)

# print(p[5]/np.pi * 180)
# print(p[10]/np.pi * 180)

test = fits.getheader(r'/test_data/test1.fits')
test_data = fits.getdata(r'/test_data/test1.fits')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
for i, ax_item in enumerate([ax1, ax2, ax3]):
    ax_item.imshow(test_data.sum(axis=i))
plt.show()

test_data1 = test_data.transpose(2,1,0)
for i, ax_item in enumerate([ax1, ax2, ax3]):
    ax_item.imshow(test_data1.sum(axis=i))
plt.show()

m16 = fits.open(r'/test_data/M16 data/hdu0_mosaic_L_3D_sigma_04.fits')

data_hdu = fits.PrimaryHDU(test_data1, header=m16[0].header)
fits.HDUList([data_hdu]).writeto('m_s_fits_name.fits', overwrite=True)  # 保存文件

data_hdu = fits.PrimaryHDU(data1, header=m16[0].header)
fits.HDUList([data_hdu]).writeto('data1.fits', overwrite=True)


exc_data1 = data1.transpose(2,1,0)

print(data1[10,12,13])

print(exc_data1[13,12,10])

print(data1[18,21,16])

print(exc_data1[16,21,18])

data_zhou = fits.getdata(r'/test_data/test_exc.fits')
Xin, Yin, Vin = np.mgrid[0:100, 0:100, 0:100]
X = np.vstack([Vin.flatten(), Yin.flatten(), Xin.flatten()]).T
Y = data_zhou.flatten()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
for i, ax_item in enumerate([ax1, ax2, ax3]):
    ax_item.imshow(data_zhou.sum(axis=i))
plt.show()

params = A - 1
params = A[:8]

ind = np.where(Y > 0.5)[0]
X2 = X[ind, ...]
Y2 = Y[ind, ...]

p = fit_gauss_3d(X2,Y2,params + 20)
print(p)
print((p[5]/np.pi * 180) % 180)