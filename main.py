import os

import cv2
import numpy as np

from format_pcd import *
from frame_cmask import make_final_cmask
from pose import get_sync_timestamps
from utils.vis_pcd import load_PCD

NUM_CAMERAS = 4

# need to ensure camera_{k} is same as camera position definition
cameras = np.array([[], [], [], []])
# cameras/300*224

# dir for gt, cmask
gt_dir = f"images/gt"
cmask_dir = f"images/cmask"
os.makedirs(gt_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# Timestamp-based frame alignment
ply_paths = [f"realsense_data/camera_{view}/ply" for view in NUM_CAMERAS]
timestamps = get_sync_timestamps(
    ply_paths, filetype="ply", eps=1000
).T  # timestamp x view

# Precompute
picked_points = [np.load(f"constants/picked_points_{view}.npy") for view in NUM_CAMERAS]
DOI_planes = [define_plane(picked_points[view]) for view in NUM_CAMERAS]
affice_matrices = [affine_matrix(picked_points[view]) for view in NUM_CAMERAS]

for row in timestamps:
    views = []
    for view in NUM_CAMERAS:
        ply_path = os.path.join(ply_paths[view], row[view] + ".ply")
        pcd = load_PCD(ply_path)
        cropped_points = crop_PCD(pcd, *DOI_planes[view])
        points = flatten(cropped_points)
        views.append(points)
    gt, cmask = make_final_cmask(points, cameras=cameras)
    gt_path = os.path.join(gt_dir, row[0] + ".jpg")
    cmask_path = os.path.join(cmask_dir, row[0] + ".jpg")
    cv2.imwrite(gt_path, gt * 255)
    cv2.imwrite(cmask_path, cmask * 255)
