import numpy as np
from typing import List

def mean_pixel_red(img: np.ndarray):
    return np.mean(img[:,:,0])