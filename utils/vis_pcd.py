import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R


def colormap(ply_path):
    depth = np.load(ply_path) / 1000.0  # from mm to meters
    depth_trunc = 8.0
    depth[depth > depth_trunc] = np.nan

    # Plot truncated depth
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap="jet", vmin=0, vmax=depth_trunc)
    plt.colorbar(label="Depth (m)")
    plt.title(f"Depth Visualization")
    plt.axis("off")
    plt.show()


def load_PCD(ply_path, show_PCD=False, show_ranges=False):
    pcd = o3d.io.read_point_cloud(ply_path)

    if show_PCD:
        o3d.visualization.draw_geometries([pcd], window_name="PCD")
        print("Press 'q' to Exit.")

    if show_ranges:
        points = np.asarray(pcd.points)
        print(f"Total Points: {len(points)}")
        print("X min:", points[:, 0].min())
        print("X max:", points[:, 0].max())
        print("Y min:", points[:, 1].min())
        print("Y max:", points[:, 1].max())
        print("Z min:", points[:, 2].min())
        print("Z max:", points[:, 2].max())

    return pcd


def select_points_PCD(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    print("Select Points with Shift+LeftClick. Press 'q' to Exit.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_points_indices = vis.get_picked_points()
    picked_points = np.asarray(pcd.points)[picked_points_indices]
    print("Picked Points:\n", picked_points)

    return picked_points


def rotate_PCD(pcd, angles, inverse=False):
    # angles = [roll, pitch, yaw] in degrees
    r = R.from_euler("xyz", angles, degrees=True)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()

    if inverse:
        T = np.linalg.inv(T)

    return pcd.transform(T)


def filter_PCD(pcd, crop=True, range={}, voxel_size=0.03, show_PCD=False):
    points = np.asarray(pcd.points)

    if crop:
        # range = {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}
        if range["xmin"]:
            points = points[points[:, 0] >= range["xmin"]]
        if range["xmax"]:
            points = points[points[:, 0] <= range["xmax"]]
        if range["ymin"]:
            points = points[points[:, 1] >= range["ymin"]]
        if range["ymax"]:
            points = points[points[:, 1] <= range["ymax"]]
        if range["zmin"]:
            points = points[points[:, 2] >= range["zmin"]]
        if range["zmax"]:
            points = points[points[:, 2] <= range["zmax"]]
        print(f"Remaining Points: {len(points)}")

    # # Filter PCD
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)
    filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=voxel_size)
    filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=1.5
    )

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


if __name__ == "__main__":
    ply_path = ""
    save_dir = "../constants/picked_points.npy"
    picked_points = select_points_PCD(ply_path)
    np.save(save_dir, picked_points)
