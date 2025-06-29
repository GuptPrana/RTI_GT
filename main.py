import os

import cv2
import numpy as np

from format_pcd import *
from frame_cmask import make_final_cmask
from pose import get_sync_timestamps
from utils.vis_pcd import load_PCD

if __name__ == "__main__":
    num_cameras = 4
    image_size = 224
    frame_size = 300
    data_folder = "realsense_data"

    ### Must ensure camera_{k} is k-th entry in cameras
    cameras = np.array([[-30, 112], [224 + 30, 112], [112, -30], [112, 224 + 30]])
    cameras = cameras * image_size / frame_size

    # dir for gt, cmask
    gt_dir = f"images/{data_folder}/gt"
    cmask_dir = f"images/{data_folder}/cmask"
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Timestamp-based frame alignment
    ply_paths = [f"{data_folder}/camera_{view}/ply" for view in num_cameras]
    timestamps = get_sync_timestamps(
        ply_paths, filetype="ply", eps=1000
    ).T  # timestamp x view

    # Precompute
    picked_points = [
        np.load(f"constants/picked_points_{view}.npy") for view in num_cameras
    ]
    DOI_planes = [define_plane(picked_points[view]) for view in num_cameras]
    affice_matrices = [affine_matrix(picked_points[view]) for view in num_cameras]

    for row in timestamps:
        views = []
        for view in num_cameras:
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
