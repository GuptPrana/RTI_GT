import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from format_pcd import *
from frame_cmask import make_final_cmask
from pose import get_sync_timestamps
from utils.vis_pcd import load_PCD


@dataclass
class Config:
    datafolder: str
    num_cameras: int = 4
    image_size: int = 224
    gt_dir: str = "images/gt"
    cmask_dir: str = "images/cmask"
    input_filetype: str = "ply"
    output_filetype: str = "jpg"
    cameras: Optional[np.ndarray] = None
    dst_pts: Optional[np.ndarray] = None
    paths: Optional[list] = None


def align_timestamps(config):
    # Timestamp-based frame alignment
    paths = [
        f"{config.datafolder}/camera_{view}/{config.input_filetype}"
        for view in range(config.num_cameras)
    ]
    config.paths = paths
    timestamps = get_sync_timestamps(
        paths, filetype=config.filetype, eps=1000
    ).T  # timestamp x view
    return timestamps


def create_dataset(config, timestamps, picked_points):
    # precompute
    DOI_planes = [
        define_plane(picked_points[view]) for view in range(config.num_cameras)
    ]
    affine_matrices = [
        affine_matrix(picked_points[view], config.dst_pts)
        for view in range(config.num_cameras)
    ]

    os.makedirs(config.gt_dir, exist_ok=True)
    os.makedirs(config.cmask_dir, exist_ok=True)

    for row in timestamps:
        views = []
        for view in range(config.num_cameras):
            ply_path = os.path.join(
                config.paths[view], row[view] + "." + config.input_filetype
            )
            pcd = load_PCD(ply_path)
            cropped_points = crop_PCD(pcd, *DOI_planes[view])
            points = flatten(affine_matrices[view], cropped_points, dst_pts.max())
            views.append(points)
        gt, cmask = make_final_cmask(points, cameras=config.cameras)
        gt_path = os.path.join(config.gt_dir, row[0] + "." + config.output_filetype)
        cmask_path = os.path.join(
            config.cmask_dir, row[0] + "." + config.output_filetype
        )
        cv2.imwrite(gt_path, gt * 255)
        cv2.imwrite(cmask_path, cmask * 255)


if __name__ == "__main__":
    # Camera positions [x, y] in global coordinates *= image_size/frame_size (e.g. 224 pixels / 3 meters)
    # Must ensure camera_{k} is k-th entry in cameras
    cameras = np.array([[], [], [], []])
    # 3D Positions of DOI Corners [x, y, z] in meters
    dst_pts = np.array([[0, 0, 1.5], [3, 0, 1.5], [0, 3, 1.5], [3, 3, 1.5]])

    config = Config(
        datafolder="realsense_data",
        gt_dir="images/gt",
        cmask_dir="images/cmask",
        cameras=cameras,
        dst_pts=dst_pts,
    )

    timestamps = align_timestamps(config)
    picked_points = [
        np.load(f"constants/picked_points_{view}.npy")
        for view in range(config.num_cameras)
    ]
    create_dataset(config, timestamps, picked_points)
