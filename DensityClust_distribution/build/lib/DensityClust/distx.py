import numpy as np


def distx(kc1, kc2, xx):
    if xx.shape[1] == 3:
        distance = np.sqrt((xx[kc1, 0] - xx[kc2, 0]) ** 2 + (xx[kc1, 1] - xx[kc2, 1]) ** 2 +
                           (xx[kc1, 2] - xx[kc2, 2]) ** 2)
    else:
        distance = np.sqrt((xx[kc1, 0] - xx[kc2, 0]) ** 2 + (xx[kc1, 1] - xx[kc2, 1]) ** 2)
    return distance
