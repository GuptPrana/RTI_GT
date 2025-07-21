import os
import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from format_pcd import *
from frame_cmask import make_final_cmask

from pose import get_sync_timestamps
from utils.npy_to_ply import npy_to_ply


@dataclass
class Config:
    datafolder: str
    object_count: int
    save_raw_pcd: bool = False  # raw .ply file
    save_DOI_pcd: bool = False  # cropped DOI points
    see_2D_points: bool = False  # post flatten()
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
    return None
    timestamps = get_sync_timestamps(
        datapaths, filetype=config.filetype, eps=1000
    ).T  # np.array(timestamp X view)
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
    affine_matrices = []
    DOI_planes = []
    for view in range(config.num_cameras):
        picked_points = np.load(config.picked_points_paths[view])
        corners, vh, polygon, centroid = define_plane(picked_points)
        DOI_planes.append([vh, polygon, centroid])
        affine_matrices.append(affine_matrix(corners, config.dst_pts))
    return DOI_planes, affine_matrices


def create_dataset(config, timestamps):
    os.makedirs(os.path.join(config.gt_dir, config.datafolder), exist_ok=True)
    os.makedirs(os.path.join(config.cmask_dir, config.datafolder), exist_ok=True)
    DOI_planes, affine_matrices = precompute_constants(config)

    rows = tqdm(timestamps)
    for row in rows:
        all_points = []
        for view in range(config.num_cameras):
            intrinsics = config.intrinsics_list[view]
            # fix depth to npy
            npypath = os.path.join(
                config.datapaths[view], row[view] + "." + config.input_filetype
            )

            pcd = npy_to_ply(npypath, intrinsics, save=config.save_raw_pcd)
            cropped_points = crop_PCD(pcd, *DOI_planes[view])  # 3D to 2D points
            points = flatten(
                affine_matrices[view],
                cropped_points,
                config.DOI_size,
                buffer=config.buffer,
            )
            all_points.append(points)

        if config.see_2D_points:
            plt.figure()
            plt.scatter(*np.vstack(all_points).T, marker="o", s=1)
            plt.title("Flattened Points")
            plt.xlim((0, 224))
            plt.ylim((0, 224))
            plt.show()

        gt, cmask = make_final_cmask(
            all_points, cameras=config.cameras, object_count=config.object_count
        )
        gt_path = os.path.join(
            config.gt_dir, config.datafolder, row[0] + "." + config.output_filetype
        )
        cmask_path = os.path.join(
            config.cmask_dir, config.datafolder, row[0] + "." + config.output_filetype
        )
        # cv2 and npy set (0, 0) at top left by default.
        cv2.imwrite(gt_path, np.flipud(gt*255))
        cv2.imwrite(cmask_path, np.flipud(cmask*255))
        rows.set_description(f"Prepared GT for {config.datafolder}/{row[0]}")


if __name__ == "__main__":
    # Camera positions [x, y] in global coordinates *= image_size/frame_size (e.g. 224 pixels / 3 meters)
    # Must ensure camera_{k} is k-th entry in cameras
    cameras = np.array(
        [[279, 112], [112, 280], [-45, 112], [112, -63]]
    )  # 0.73, 0.75, -0.60, -0.85
    # 3D Positions of DOI Corners [x, y] #, z] in meters
    dst_pts = np.array([[3, 0], [0, 0], [0, 3], [3, 3]])

    config = Config(
        datafolder="realsense_data_306_b",
        gt_dir=os.path.join("images", "gt"),
        cmask_dir=os.path.join("images", "cmask"),
        cameras=cameras,
        dst_pts=dst_pts,
        object_count=2,  # objects in DOI
    )

    config.see_2D_points = 1
    timestamps = align_timestamps(config)
    timestamps = [
        ["30154743160105", "30154743170676", "30154716339960", "30154716362644"]
    ]

    config.picked_points_paths = [
        os.path.join("constants", f"picked_points_{view}.npy")
        for view in range(config.num_cameras)
    ]
    intrinsics_paths = [
        os.path.join(
            config.datafolder, f"camera_{view}", "intrinsics", "intrinsics.json"
        )
        for view in range(config.num_cameras)
    ]
    config.intrinsics_list = prepare_intrinsics(intrinsics_paths)
    create_dataset(config, timestamps)
