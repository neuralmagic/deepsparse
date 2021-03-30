'''
Example created by Adriano A. Santos (https://github.com/adrianosantospb).
'''

import os
import cv2
import numpy as np
import torch

# Letterbox function
def letterbox(image, desired_size=416):
    old_size = image.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    
    return new_img

# pre-processing fuction
def preprocessing(image, image_size):
    # Padded resize
    img = letterbox(image,image_size)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0).numpy()
    
    return img