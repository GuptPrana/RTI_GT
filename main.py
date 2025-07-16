import os
import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from format_pcd import *
from frame_cmask import make_final_cmask
from pose import get_sync_timestamps
from utils.vis_pcd import load_PCD
from utils.npy_to_ply import npy_to_ply


@dataclass
class Config:
    datafolder: str
    object_count: int
    save_raw_pcd: bool = False  # raw .ply file
    save_DOI_pcd: bool = False  # cropped DOI points
    num_cameras: int = 4
    image_size: int = 224
    DOI_size: int = 3
    buffer: int = 15
    gt_dir: str = "images/gt"
    cmask_dir: str = "images/cmask"
    input_filetype: str = "npy"
    output_filetype: str = "jpg"
    intrinsics_list: Optional[list] = None
    cameras: Optional[np.ndarray] = None
    dst_pts: Optional[np.ndarray] = None
    picked_points_paths: Optional[list] = None
    datapaths: Optional[list] = None


def align_timestamps(config):
    # Timestamp-based frame alignment
    datapaths = [
        f"{config.datafolder}/camera_{view}/{config.input_filetype}"
        for view in range(config.num_cameras)
    ]
    config.datapaths = datapaths
    timestamps = get_sync_timestamps(
        datapaths, filetype=config.filetype, eps=1000
    ).T  # timestamp x view
    return timestamps


def prepare_intrinsics(intrinsics_paths):
    intrinsics_list = []
    for path in intrinsics_paths:
        with open(path, "r") as f:
            intrinsics = json.load(f)
        intrinsics_list.append(
            [
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["ppx"],
                intrinsics["ppy"],
            ]
        )
    return intrinsics_list


def precompute_constants(config):
    # precompute
    picked_points = [np.load(path) for path in config.picked_points_paths]
    DOI_planes = [
        define_plane(picked_points[view]) for view in range(config.num_cameras)
    ]
    affine_matrices = [
        affine_matrix(picked_points[view], config.dst_pts)
        for view in range(config.num_cameras)
    ]
    return DOI_planes, affine_matrices


def create_dataset(config, timestamps):
    os.makedirs(config.gt_dir, exist_ok=True)
    os.makedirs(config.cmask_dir, exist_ok=True)
    DOI_planes, affine_matrices = precompute_constants(config)

    for row in timestamps:
        all_points = []
        for view in range(config.num_cameras):
            intrinsics = config.intrinsics_list[view]
            # fix depth to npy
            npypath = os.path.join(
                config.datapaths[view], row[view] + "." + config.input_filetype
            )

            pcd = npy_to_ply(npypath, intrinsics, save=config.save_raw_pcd)
            cropped_pcd = crop_PCD(pcd, *DOI_planes[view])
            if config.save_DOI_pcd:
                base_dir = os.path.join(
                    os.path.split(config.datapaths[view]).pop(-1), "ply"
                )
                os.makedirs(base_dir, exist_ok=True)
                ply_path = os.path.join(base_dir, row[view] + ".ply")
                pcd = flatten(
                    affine_matrices[view],
                    cropped_pcd.points,
                    config.DOI_size,
                    as_pcd=True,
                    ply_path=ply_path,
                )
                continue
            else:
                points = flatten(
                    affine_matrices[view],
                    cropped_pcd.points,
                    config.DOI_size,
                    buffer=config.buffer,
                )
            all_points.append(points)

        gt, cmask = make_final_cmask(
            all_points, cameras=config.cameras, object_count=config.object_count
        )
        gt_path = os.path.join(config.gt_dir, row[0] + "." + config.output_filetype)
        cmask_path = os.path.join(
            config.cmask_dir, row[0] + "." + config.output_filetype
        )
        cv2.imwrite(gt_path, gt * 255)
        cv2.imwrite(cmask_path, cmask * 255)


if __name__ == "__main__":
    # Camera positions [x, y] in global coordinates *= image_size/frame_size (e.g. 224 pixels / 3 meters)
    # Must ensure camera_{k} is k-th entry in cameras
    cameras = np.array(
        [[279, 112], [112, 280], [-45, 112], [112, -63]]
    )  # 0.73, 0.75, -0.60, -0.85
    # 3D Positions of DOI Corners [x, y, z] in meters
    dst_pts = np.array([[3, 0, 1.5], [0, 0, 1.5], [0, 3, 1.5], [3, 3, 1.5]])

    config = Config(
        datafolder="realsense_data",
        gt_dir=os.path.join("images", "gt"),
        cmask_dir=os.path.join("images", "cmask"),
        cameras=cameras,
        dst_pts=dst_pts,
        object_count=2,  # objects in DOI
    )

    timestamps = align_timestamps(config)
    config.picked_points_paths = [
        os.path.join("constants", f"picked_points_{view}.npy")
        for view in range(config.num_cameras)
    ]
    intrinsics_paths = [
        os.path.join(
            config.data_folder, f"camera_{view}", "intrinsics", "intrinsics.json"
        )
        for view in range(config.num_cameras)
    ]
    config.intrinsics_list = prepare_intrinsics(intrinsics_paths)
    create_dataset(config, timestamps)
