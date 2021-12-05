import os
from DensityClust.densityCluster_3d import densityCluster_3d, densityCluster_2d
from DensityClust.clustring_subfunc import *
# https://stackoverflow.com/questions/16490261/python-pandas-write-dataframe-to-fixed-width-file-to-fwf


def localDenCluster(data_name, para=None, mask_name=None, outcat_name=None, outcat_wcs_name=None):
    """
    LDC algorithm
    :param data_name: 待检测数据的路径(str)，fits文件
    :param para: 算法参数，dict
        para.rhomin: Minimum density
        para.deltamin: Minimum delta
        para.v_min: Minimum volume
        para.rms: The noise level of the data, used for data truncation calculation
        para.sigma: 密度估计的窗口
    :param mask_name: 掩模数据的保存路径(str)
    :param outcat_name: 基于像素单位的核表保存路径(str)
    :param outcat_wcs_name: 基于wcs的核表保存路径(str)
    :return:
    """
    data_port_dir = data_name.replace('.fits', '')  # 默认保存路径
    if para is None:
        para = {"gradmin": 0.01, "rhomin": 0.8, "deltamin": 4, "v_min": 27, "rms": 0.46, "dc": 0.6, "is_plot": 0}

    data = fits.getdata(data_name)
    data[np.isnan(data)] = 0
    data_ndim = data.ndim
    if data_ndim == 2:
        print("2d")
        outcat, mask, out = densityCluster_2d(data, para)

    elif data_ndim == 3:
        print("3d data")
        t0 = time.time()
        outcat, mask, out = densityCluster_3d(data, para)
        t1 = time.time()
        print('LDC time: %0.2f' % (t1-t0))
    else:
        outcat, mask, out = None, None, None
        print(data.shape)

    if outcat_name is None:
        if not os.path.exists(data_port_dir):
            os.mkdir(data_port_dir)
        outcat_name = os.path.join(data_port_dir, 'LDC_outcat.txt')
    if mask_name is None:
        if not os.path.exists(data_port_dir):
            os.mkdir(data_port_dir)
        mask_name = os.path.join(data_port_dir, 'LDC_mask.fits')
    if outcat_wcs_name is None:
        if not os.path.exists(data_port_dir):
            os.mkdir(data_port_dir)
        outcat_wcs_name = os.path.join(data_port_dir, 'LDC_outcat_wcs.txt')

    if os.path.isfile(mask_name):
        os.remove(mask_name)
        fits.writeto(mask_name, mask)
    else:
        fits.writeto(mask_name, mask)

    save_outcat(outcat_name=outcat_name, outcat=outcat)

    outcat = pd.read_csv(outcat_name, sep='\t')
    data_wcs = get_wcs(data_name)
    outcat_wcs = change_pix2word(data_wcs, outcat)

    save_outcat(outcat_name=outcat_wcs_name, outcat=outcat_wcs)


if __name__ == '__main__':
    pass

