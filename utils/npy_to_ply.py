import json
import os

import cv2
import numpy as np
import open3d as o3d

from utils.vis_pcd import rotate_PCD, filter_PCD


def undistort(npy_path, distortion_coeffs):
    pass  # coeffs = 0.0


def npy_to_ply(npy_path, intrinsics, ply_path=None, rough_crop_pcd=False, visualize=False, save=False, **kwargs):
    depth = np.load(npy_path).astype(np.uint16)

    # Filtering
    depth = cv2.bilateralFilter(
        depth.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50
    )
    depth = depth.astype(np.uint16)
    depth_o3d = o3d.geometry.Image(depth)

    # Optional, Can Fuse with RGB
    gray = np.uint8((depth / np.max(depth)) * 255)
    gray3d = o3d.geometry.Image(np.stack([gray] * 3, axis=-1))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        gray3d,
        depth_o3d,
        depth_scale=1000.0,  # mm to meters
        depth_trunc=7.25,  # *trunc
        convert_rgb_to_intensity=False,
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(*intrinsics)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if rough_crop_pcd:
        pcd = rotate_PCD(pcd, angles=kwargs['angles'])
        pcd = filter_PCD(pcd, crop=True, ranges=kwargs['ranges'])
        pcd = rotate_PCD(pcd, angles=kwargs['angles'], inverse=True)

    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Preview Point Cloud")
        print("Press 'q' to Exit.")

    if save:
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"Saved point cloud to: {ply_path}")

    return pcd


if __name__ == "__main__":
    cam_view = 0
    data_folder = "realsense_data_306_b"

    npy_dir = os.path.join(data_folder, f"camera_{cam_view}", "npy")
    ply_dir = os.path.join(data_folder, f"camera_{cam_view}", "ply")
    os.makedirs(ply_dir, exist_ok=True)
    intrinsics_path = os.path.join(
        data_folder, f"camera_{cam_view}", "intrinsics", "intrinsics.json"
    )

    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
    intrinsics_info = [
        intrinsics["width"],
        intrinsics["height"],
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["ppx"],
        intrinsics["ppy"],
    ]
    distortion_coeffs = intrinsics["coeffs"]

    filelist = os.listdir(npy_dir)
    startidx = filelist.index("30154743090546.npy")  # 0
    endidx = filelist.index("30162940278584.npy")  # -1
    for npy in filelist[startidx : endidx + 1]:
        if npy.endswith(".npy"):
            name = npy.split(".")[0]
            npy_path = os.path.join(npy_dir, npy)
            ply_path = os.path.join(ply_dir, name + ".ply")
            npy_to_ply(npy_path, ply_path, intrinsics_info, verbose=1)
            break  # for testing
