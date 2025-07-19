import os

import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R


def colormap(npy_path, min_depth=-1.0, max_depth=8.0):
    depth = np.load(npy_path) / 1000.0  # from mm to meters
    depth[depth > max_depth] = np.nan
    depth[depth < min_depth] = np.nan

    # Plot truncated depth
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap="jet", vmin=min_depth, vmax=max_depth)
    plt.colorbar(label="Depth (m)")
    plt.title(f"Depth Visualization")
    plt.axis("off")
    plt.show()


def show_ranges_PCD(points):
    print(f"Total Points: {len(points)}")
    print("X min:", points[:, 0].min())
    print("X max:", points[:, 0].max())
    print("Y min:", points[:, 1].min())
    print("Y max:", points[:, 1].max())
    print("Z min:", points[:, 2].min())
    print("Z max:", points[:, 2].max())
    return


def load_PCD(ply_path, show_PCD=False, show_ranges=False):
    pcd = o3d.io.read_point_cloud(ply_path)

    if show_PCD:
        o3d.visualization.draw_geometries([pcd], window_name="PCD")
        print("Press 'q' to Exit.")

    if show_ranges:
        points = np.asarray(pcd.points)
        show_ranges_PCD(points)

    return pcd


def select_points_PCD(ply):
    print("Select Points with Shift+LeftClick. Press 'q' to Exit.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(ply)
    vis.run()
    vis.destroy_window()
    picked_points_indices = vis.get_picked_points()
    picked_points = np.asarray(ply.points)[picked_points_indices]
    print("Picked Points:\n", picked_points)

    if save:
        np.save(save_path, picked_points)

    return


def rotate_PCD(pcd, angles, inverse=False, show_ranges=False, only_matrix=False):
    # angles = [roll, pitch, yaw] in degrees
    r = R.from_euler("xyz", angles, degrees=True)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()

    if inverse:
        T = np.linalg.inv(T)

    if only_matrix:
        return T

    pcd = pcd.transform(T)
    if show_ranges:
        show_ranges_PCD(np.asarray(pcd.points))

    return pcd


def filter_PCD(pcd, crop=False, ranges={}, voxel_size=0.03, show_PCD=False):
    points = np.asarray(pcd.points)

    if crop:
        # ranges = {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}
        if "xmin" in ranges.keys():
            points = points[points[:, 0] >= ranges["xmin"]]
        if "xmax" in ranges.keys():
            points = points[points[:, 0] <= ranges["xmax"]]
        if "ymin" in ranges.keys():
            points = points[points[:, 1] >= ranges["ymin"]]
        if "ymax" in ranges.keys():
            points = points[points[:, 1] <= ranges["ymax"]]
        if "zmin" in ranges.keys():
            points = points[points[:, 2] >= ranges["zmin"]]
        if "zmax" in ranges.keys():
            points = points[points[:, 2] <= ranges["zmax"]]
        print(f"Remaining Points: {len(points)}")

    # # Filter PCD
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)
    filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
    filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=1.5
    )
    print(len(filtered_pcd.points))

    if show_PCD:
        # Red, Green, Blue = X, Y, Z
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # filtered_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries(
            [filtered_pcd, coord_frame],
            window_name="Filtered PCD",
            point_show_normal=False,
        )
        print("Press 'q' to Exit.")

    return filtered_pcd


def crop_PCD(pcd, picked_points, eps=0.01, show_PCD=False):
    # picked_points = np.array([Nx3]), Corners of DOI Plane.
    centroid = picked_points.mean(axis=0)
    centered = picked_points - centroid
    _, _, vh = np.linalg.svd(centered)

    normal = vh[2]  # Normal = smallest singular vector
    u = vh[0]  # X
    v = vh[1]  # Y

    # Project pcd.points to (u, v) plane
    points = np.asarray(pcd.points)
    rel_points = points - centroid
    proj_u = rel_points @ u
    proj_v = rel_points @ v
    proj_2d = np.vstack((proj_u, proj_v)).T

    corner_rel = picked_points - centroid
    corner_uv = np.stack([corner_rel @ u, corner_rel @ v], axis=1)

    # Check inclusion
    polygon = Path(corner_uv)
    mask = polygon.contains_points(proj_2d)
    point_plane_dist = np.abs(rel_points @ normal)
    final_mask = mask & (point_plane_dist < eps)
    cropped_pcd = pcd.select_by_index(np.where(final_mask)[0])

    if show_PCD:
        o3d.visualization.draw_geometries([cropped_pcd])
        print("Press 'q' to Exit.")

    return cropped_pcd


def main():
    cam_view = 1
    data_folder = "realsense_data"

    ply_dir = os.path.join(data_folder, f"camera_{cam_view}", "ply")
    ply_path = os.path.join(ply_dir, os.listdir(ply_path)[0])
    ply = load_PCD(ply_path)  # , show_PCD=True)

    # Cropping out DOI for easier point selection
    angles = [-20, 0, -2]
    ranges = {
        "ymin": -1.4,
        "ymax": 0.8,
        "zmin": -7,
        "xmin": -3,
        "xmax": 3,
    }

    ply = rotate_PCD(ply, angles=angles, show_ranges=True)
    ply = filter_PCD(ply, crop=True, ranges=ranges, show_PCD=True)
    # Must to keep alignment of picked_points
    ply = rotate_PCD(ply, angles=angles, inverse=True)

    save_dir = os.path.join("constants", f"picked_points_{cam_view}.npy")
    picked_points = select_points_PCD(ply)

    if picked_points.size:
        input_order = input(
            "Enter ordered indices (e.g. 0, 2, 6, 3) or leave blank to skip: "
        ).strip()
        try:
            if input_order:
                order = [int(i.strip()) for i in input_order.split(",")]
                if max(order) < len(picked_points):
                    picked_points = picked_points[order]
                else:
                    print("Invalid indices.")
            else:
                print("No indices entered.")
        except Exception as e:
            print(f"Error: {e}.")

        np.save(save_dir, picked_points)
        print(f"Saved {len(picked_points)} selected points to {save_dir}.")


if __name__ == "__main__":
    main()
