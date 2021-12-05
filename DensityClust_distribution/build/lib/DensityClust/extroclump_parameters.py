import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from LDC_version.DensityClust import tools, get_xx, make_plot


def make_tri_plot(data):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(8, 6))
    ax0.imshow(data.sum(axis=0), cmap='gray',)
    ax1.imshow(data.sum(axis=1), cmap='gray')
    ax2.imshow(data.sum(axis=2), cmap='gray')

    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def extroclump_parameters(clustInd, centInd, data):
    """
    计算根据LDC得到的聚类的云核参数

    :param clustInd: 聚类的编号，同一个类用一个相同的数字标记
    :param centInd: centInd: centInd = [centIndex, clust_id] 代表聚类中心点在data中的索引以及聚类的编号(ID)
    :param data: 3D data
    :return:
    output:
        Peak1,Peak2,Peak3,Cen1,Cen2,Cen3,Size1,Size2,Size3,Peak,Sum,Volume
        Size1, Size2, Size3  为半高全宽(FWHM)
    out: clumps data
    mask: mask
    """
    
    NClust = centInd.shape[0]
    xx = get_xx.get_xyz(data)  # xx: 3D data coordinates  xx坐标系的坐标原点为1
    dim = data.ndim
    mask = np.zeros_like(data, dtype=np.int)
    out = np.zeros_like(data, dtype=np.float)

    clustSum = np.zeros([NClust, 1], dtype=np.float)
    clustVolume = np.zeros([NClust, 1], dtype=np.int)
    clustPeak = np.zeros([NClust, 1], dtype=np.float)
    clump_Cen = np.zeros([NClust, dim], dtype=np.float)
    clustSize = np.zeros([NClust, dim], dtype=np.float)
    if dim == 3:
        clump_Peak = xx[centInd[:, 0], :]   # xx坐标系的坐标原点为1

        for i in range(NClust):

            cl_1_index_ = xx[np.where(clustInd == (centInd[i, 1]) + 1)[0], :] - 1  # -1 是为了在data里面用索引取值(从0开始)
            # clustInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
            clustNum = cl_1_index_.shape[0]

            cl_i = np.zeros_like(data, np.int)
            for j, item in enumerate(cl_1_index_):
                cl_i[item[2], item[1], item[0]] = 1
            # 形态学处理
            L = measure.label(cl_i)   # Labeled input image. Labels with value 0 are ignored.
            STATS = measure.regionprops(L)

            Ar_sum = []
            for region in STATS:
                coords = region.coords  # 经过验证，坐标原点为0
                temp = 0
                for j, item in enumerate(coords):
                    temp += data[item[0], item[1], item[2]]
                Ar_sum.append(temp)
            Ar = np.array(Ar_sum)
            ind = np.where(Ar == Ar.max())[0]
            L[L != ind + 1] = 0
            cl_i = L / (ind[0] + 1)
            coords_maxAr = STATS[ind[0]].coords  # 最大的连通域对应的坐标

            clump_i_ = np.zeros(coords_maxAr.shape[0])
            for j, item in enumerate(coords_maxAr):
                clump_i_[j] = data[item[0], item[1], item[2]]

            clump_i = data * cl_i
            out = out + clump_i
            mask = mask + cl_i * (i + 1)
            clustsum = clump_i_.sum() + 0.0001  # 加一个0.0001 防止分母为0
            coords_maxAr = coords_maxAr[:, [2, 1, 0]]  # 和xx坐标系保持一致
            clump_Cen[i, :] = np.matmul(clump_i_, coords_maxAr) / clustsum  # 这里的坐标，坐标原点为0
            clustVolume[i, 0] = clustNum
            clustSum[i, 0] = clustsum
            clustPeak[i, 0] = data[clump_Peak[i, 2] - 1, clump_Peak[i, 1] - 1, clump_Peak[i, 0] - 1]

            x_i = coords_maxAr - clump_Cen[i, :]
            clustSize[i, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                      - (np.matmul(clump_i_, x_i) / clustsum) ** 2)

    else:
        clustSum, clustVolume, clustPeak = np.zeros([NClust, 1]), np.zeros([NClust, 1]), np.zeros([NClust, 1])
        clump_Cen, clustSize = np.zeros([NClust, dim]), np.zeros([NClust, dim])
        clump_Peak = xx[centInd[:, 0], :]

        for i in range(0, NClust):

            cl_i = np.zeros_like(data, np.int)
            cl_1_index_ = xx[np.where(clustInd == (centInd[i, 1]) + 1)[0], :] - 1  # -1 是为了在data里面用索引取值(从0开始)
            clustNum = cl_1_index_.shape[0]

            for j, item in enumerate(cl_1_index_):
                # print(item)
                cl_i[item[1], item[0]] = 1  # (i + 1) 是云核的编号  从1开始

            print(i)
            L = measure.label(cl_i)  # 给连通域打标签
            STATS = measure.regionprops(L)  # 统计连通域的信息
            Ar_sum = []
            for region in STATS:
                # Ar.append(region.area)
                coords = region.coords
                coords = coords[:, [1, 0]]
                temp = 0
                for j, item in enumerate(coords):
                    temp += data[item[1], item[0]]
                Ar_sum.append(temp)
            Ar = np.array(Ar_sum)
            ind = np.where(Ar == Ar.max())[0]
            L[L != ind + 1] = 0
            cl_i = L / (ind[0] + 1)
            coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标

            coords = coords[:, [1, 0]]
            clump_i_ = np.zeros(coords.shape[0])
            for j, item in enumerate(coords):
                clump_i_[j] = data[item[1], item[0]]

            clustsum = sum(clump_i_) + 0.0001  # 加一个0.0001 防止分母为0
            clump_Cen[i, :] = np.matmul(clump_i_, coords) / clustsum
            clustVolume[i, 0] = clustNum
            clustSum[i, 0] = clustsum
            clustPeak[i, 0] = data[clump_Peak[i, 1] - 1, clump_Peak[i, 0] - 1]
            x_i = coords - clump_Cen[i, :]
            clustSize[i, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                      - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
            clump_i = data * cl_i
            out = out + clump_i
            mask = mask + cl_i * (i + 1)

    clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
    id_clumps = np.array([item + 1 for item in range(NClust)], np.int).T
    id_clumps = id_clumps.reshape([NClust, 1])

    outcat = np.column_stack((id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))

    return outcat, mask, out


# def extroclump_parameters(centNum, xxyy, co_real, centInd, origin_data):
#     dim = len(origin_data.shape)
#     NCLUST = centNum.shape[0]
#     [xres, yres] = origin_data.shape
#     clustSum = np.zeros(NCLUST)
#     clustVolume = np.zeros(NCLUST)
#     clustPeak = np.zeros(NCLUST)
#     clumpCenter = np.zeros([NCLUST, dim])
#     clustSize = np.zeros([NCLUST, dim])
#
#     clumpPeakIndex = xxyy[centInd]
#     describe = {}
#     for i in range(NCLUST):
#         clust_coords = xxyy[co_real == centNum[i]]
#         clustSum[i] = origin_data[clust_coords[:, 0], clust_coords[:, 1]].sum()
#         clustVolume[i] = clust_coords.shape[0]
#         clustPeak[i] = origin_data[clumpPeakIndex[i][0], clumpPeakIndex[i][1]]
#         x_center = np.sum((clust_coords[:, 0] + 1) * origin_data[clust_coords[:, 0], clust_coords[:, 1]]) / clustSum[
#             i] - 1
#         y_center = np.sum((clust_coords[:, 1] + 1) * origin_data[clust_coords[:, 0], clust_coords[:, 1]]) / clustSum[
#             i] - 1
#         clumpCenter[i] = [x_center, y_center]
#         center_delta = clust_coords - clumpCenter[i]
#         od = origin_data[clust_coords[:, 0], clust_coords[:, 1]]
#         clustSize[i] = (np.array(np.mat(od) * np.array(np.mat(center_delta)) ** 2 / clustSum[i]) \
#                         - np.array(np.mat(od) * np.mat(center_delta) / clustSum[i]) ** 2) ** (1 / 2)
#     describe['clustSum'] = clustSum
#     describe['clustVolume'] = clustVolume
#     describe['clumpPeakIndex'] = clumpPeakIndex
#     describe['clustPeak'] = clustPeak
#     describe['clumpCenter'] = clumpCenter
#     describe['clustSize'] = clustSize
#     return describe

if __name__ == '__main__':
    import astropy.io.fits as fits
    centInd = fits.getdata('centInd.fits')
    data = fits.getdata('data.fits')
    clustInd = fits.getdata('clustInd.fits')

    outcat, mask, out = extroclump_parameters(clustInd, centInd, data)
    outcat_name = 'data.fits'.replace('.fits', '_LDC1.txt')
    tools.save_outcat(outcat_name=outcat_name, outcat=outcat)
    make_plot.make_plot(outcat_name, out)
    mask1 = mask.copy()
    mask1[mask1!=67] = 0
    make_plot.make_plot(outcat_name, out * mask1 / 67)
