import json
import os

import cv2
import numpy as np
import open3d as o3d


def npy_to_ply(npy_path, ply_path, intrinsics, depth_trunc=7.25, visualize=False):
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
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    intrinsics = o3d.camera.PinholeCameraIntrinsic(*intrinsics)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name="Preview Point Cloud")
        print("Press 'q' to Exit.")

    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud to: {ply_path}")

    return


if __name__ == "__main__":
    cam_view = 1
    npy_dir = f"../realsense_data/camera_{cam_view}/depth"
    ply_dir = f"../realsense_data/camera_{cam_view}/ply"
    intrinsics_path = f"../realsense_data/camera_{cam_view}/intrinsics/intrinsics.json"
    os.makedirs(ply_dir, exist_ok=True)

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

    for npy in os.listdir(npy_dir):
        name = npy.split(".")[0]
        npy_path = os.path.join(npy_dir, npy)
        ply_path = os.path.join(ply_dir, name + ".ply")
        npy_to_ply(npy_path, ply_path, intrinsics_info)
