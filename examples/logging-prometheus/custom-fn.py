import numpy as np
from typing import List

def mean_pixel_red(img: np.ndarray):
    assert(img.shape[2] == 3)
    return np.mean(img[:,:,0])