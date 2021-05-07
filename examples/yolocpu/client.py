'''
Example created by Adriano A. Santos (https://github.com/adrianosantospb).
'''

from __future__ import print_function
import requests
import json
import cv2
import jsonpickle
from datetime import datetime
import argparse
from deepsparse.utils import arrays_to_bytes, bytes_to_arrays
import time
import numpy as np

# TODO: Refactor this code

server_address = 'http://localhost:9898'
test_url = server_address + '/api/predict'
color = (0,255,0)

parser = argparse.ArgumentParser()
opt = parser.parse_args()

VIDEO = 'video/example.avi'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# Get the divice
camera = cv2.VideoCapture(VIDEO)

# used to record the time when we processed last frame
prev_frame_time = 0
  
# used to record the time at which we processed current frame
new_frame_time = 0

fps_list = []

# Get the frames and analyze them
while True:

    # Get the current frame and the status
    status, frame = camera.read()

    # Stop condition
    key = cv2.waitKey(1) & 0xFF

    if (status == False or key == ord("q")):
        break

    img = frame.copy()

    new_frame_time = time.time()

    _, img_encoded = cv2.imencode('.jpg', img)

    # Send image to analyze and getting tags as return
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    fps_list.append(fps)
    
    outputs = bytes_to_arrays(response.content)
    
    # If you wand to see the output
    # for i, out in enumerate(outputs):
        #output 0: shape (10647, 1) - classification
        #output 1: shape (10647, 4) - bbox predictions
    #    print(f" output #{i}: shape {out.shape}")  

    cv2.imshow("Frame", frame)

print("FPS AVG: {}".format(np.mean(fps_list)))
camera.release()
cv2.destroyAllWindows()