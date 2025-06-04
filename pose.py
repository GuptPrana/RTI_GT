import os
import cv2
import torch
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

## Need Camera Matching for Scynchronizing Images. 

def apply_homography(x, y, H):
    # can access H by saved np.array 
    point = np.array([x, y, 1])
    tr = H @ point
    tr /= tr[2]
    return int(tr[0]), int(tr[1])

model2 = YOLO('yolov8x-pose-p6.pt') 
image_dir = 'realsense_data/test_seq/'
image_paths = os.listdir(image_dir)
# unique for each camera view
transform = np.load('realsense_data/transform.npy')

### Need to extend to multiperson case.
### Can simply assume 4 people max, leave cell empty if less people. 
position = []

camerapaths = ['realsense_data/cam1/', 'realsense_data/cam2/']

def getCoordinates(rootdir, transform):
    image_paths = os.listdir(rootdir)
    data = []
    for path in image_paths:
        image = cv2.imread(os.path.join(image_dir, path))
        results = model2(image)
        if len(results[0].keypoints.xy) == 0:
            continue  # No keypoints
        '''
        # For visualization
        img = results[0].plot()  # This plots the detections on the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(path + '_pose', img_rgb)
        cv2.imshow(img_rgb)
        cv2.waitKey(0)
        cv2.destroykeypoints.xy[0] 
        '''
        # Extract coordinates for left and right feet
        # Can also check if 15 and 16 are present.
        keypoints = results[0].keypoints.xy[0]
        left_foot = keypoints[15].tolist()  # left foot
        right_foot = keypoints[16].tolist()  # right foot
        x, y = (left_foot[0]+right_foot[0])/2, (left_foot[1]+right_foot[1])/2
        xT, yT = apply_homography(x, y, transform)

        data.append({
            'filename': path.strip('.')[0],
            'x': int(x),
            'y': int(y),
            'xT': int(xT),
            'yT': int(yT)
        })
    return data

dfs = []
for view in range(len(camerapaths)):
    transform = np.load(f'realsense_data/transform_{view}.npy')
    dfs.append(getCoordinates(camerapaths[view], transform))

# Joining dfs logic
# save final df to csv logic

with open('position.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(position)
