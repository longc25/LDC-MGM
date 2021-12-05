import os
import numpy as np
import astropy.io.fits as fits
import time
import pandas as pd
from astropy import wcs
from LDC_version.DensityClust import tools
from LDC_version.DensityClust.densityCluster_3d import densityCluster_3d, densityCluster_2d
from LDC_version.DensityClust.extroclump_parameters import extroclump_parameters


def get_wcs(data_name):
    """
    得到wcs信息
    :param data_name: fits文件
    :return:
    data_wcs
    """
    data_header = fits.getheader(data_name)
    keys = data_header.keys()
    key = [k for k in keys if k.endswith('4')]
    [data_header.remove(k) for k in key]
    data_header.remove('VELREF')
    data_wcs = wcs.WCS(data_header)
    return data_wcs


def change_pix2word(data_wcs, outcat):
    """
    将算法检测的结果(像素单位)转换到天空坐标系上去
    :param data_wcs: 头文件得到的wcs
    :param outcat: 算法检测核表
    :return:
    outcat_wcs
    ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume']
    """
    peak1, peak2, peak3 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1)
    clump_Peak = np.column_stack([peak1, peak2, peak3 / 1000])
    cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
    clump_Cen = np.column_stack([cen1, cen2, cen3 / 1000])
    size1, size2, size3 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30, outcat['Size3'] * 166.6])
    clustSize = np.column_stack([size1, size2, size3])
    clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

    id_clumps = []  # G017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
    for item_l, item_b, item_v in zip(cen1, cen2, cen3 / 1000):
        str_l = 'G' + ('%.03f' % item_l).rjust(7, '0')
        if item_b < 0:
            str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
        else:
            str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
        if item_v < 0:
            str_v = '-' + ('%.03f' % abs(item_v)).rjust(6, '0')
        else:
            str_v = '+' + ('%.03f' % abs(item_v)).rjust(6, '0')
        id_clumps.append(str_l + str_b + str_v)
    id_clumps = np.array(id_clumps)

    outcat_wcs = np.column_stack((id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))
    return outcat_wcs


def get_outcat_local(outcat):
    """
    返回局部区域的检测结果：
    原始图为120*120  局部区域为30-->90, 30-->90 左开右闭
    :param outcat:
    :return:
    """
    # outcat = pd.read_csv(txt_name, sep='\t')
    cen1_min = 30
    cen1_max = 90
    cen2_min = 30
    cen2_max = 90
    aa = outcat.loc[outcat['Cen1'] > cen1_min]
    aa = aa.loc[outcat['Cen1'] <= cen1_max]
    aa = aa.loc[outcat['Cen2'] > cen2_min]
    aa = aa.loc[outcat['Cen2'] <= cen2_max]
    return aa


def localDenCluster(data_name, para=None, mask_name=None, outcat_name=None, outcat_wcs_name=None):
    data_port_dir = data_name.replace('.fits', '')
    if para is None:
        para = {"gradmin": 0.01, "rhomin": 0.8, "deltamin": 4, "v_min": 27, "rms": 0.46, "sigma": 0.6, "is_plot": 0}

    if not os.path.exists(data_port_dir):
        os.mkdir(data_port_dir)

    data = fits.getdata(data_name)
    data[np.isnan(data)] = 0
    data_ndim = data.ndim
    if data_ndim == 2:
        print("2d")
        centInd, clustInd = densityCluster_2d(data, para)
    elif data_ndim == 3:
        print("3d data")
        t0 = time.time()
        centInd, clustInd = densityCluster_3d(data, para)
        t1 = time.time()
        print('LDC time: %0.2f' % (t1-t0))
    else:
        centInd, clustInd = None, None

    if centInd is not None:
        outcat, mask, out = extroclump_parameters(clustInd, centInd, data)
        if outcat_name is None:
            outcat_name = os.path.join(data_port_dir, 'LDC_outcat.txt')
        if mask_name is None:
            mask_name = os.path.join(data_port_dir, 'LDC_mask.fits')
        if outcat_wcs_name is None:
            outcat_wcs_name = os.path.join(data_port_dir, 'LDC_outcat_wcs.txt')

        if os.path.isfile(mask_name):
            os.remove(mask_name)
            fits.writeto(mask_name, mask)
        else:
            fits.writeto(mask_name, mask)

        tools.save_outcat(outcat_name=outcat_name, outcat=outcat)

        outcat = pd.read_csv(outcat_name, sep='\t')
        data_wcs = get_wcs(data_name)
        outcat_wcs = change_pix2word(data_wcs, outcat)

        tools.save_outcat(outcat_name=outcat_wcs_name, outcat=outcat_wcs)


