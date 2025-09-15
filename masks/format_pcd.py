import numpy as np
import open3d as o3d
from matplotlib.path import Path

"""
Select the corners of DOI for picked_points
in same order as dst_points below.
"""


def affine_matrix(corner_uv, dst_pts, lstsq=True, alpha=1e-3):
    N = corner_uv.shape[0]
    assert dst_pts.shape[0] == N
    src_h = np.hstack([corner_uv, np.ones((N, 1))])
    dst_h = np.hstack([dst_pts, np.ones((N, 1))])

    if lstsq:
        # A_T, residuals, _, _ = np.linalg.lstsq(src_h, dst_h, rcond=None)
        XtX = src_h.T @ src_h + alpha * np.eye(src_h.shape[1])
        XtY = src_h.T @ dst_h
        A_T = np.linalg.solve(XtX, XtY)
        return A_T.T

    return np.linalg.solve(src_h.T, dst_h.T).T


def transform_pts(points, A):
    points_ = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_ = (A @ points_.T).T
    n = A.shape[0] - 1
    transformed_points = transformed_[:, :n] / transformed_[:, [n]]  # scale
    return transformed_points


def define_plane(picked_points):
    centroid = picked_points.mean(axis=0)

    # Define u as left-to-right (P1 - P0), v as top-to-bottom (P3 - P0)
    u = picked_points[1] - picked_points[0]
    v = picked_points[3] - picked_points[0]

    u = (picked_points[1] - picked_points[0] + picked_points[2] - picked_points[3]) / 2
    v = (picked_points[3] - picked_points[0] + picked_points[2] - picked_points[1]) / 2

    if abs(u[0]) >= abs(u[1]):
        if u[0] < 0:
            u = -u
    else:
        if u[1] < 0:
            u = -u

    u /= np.linalg.norm(u)
    v -= u * np.dot(v, u)  # make v orthogonal to u
    v /= np.linalg.norm(v)
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal)
    centered = picked_points - centroid

    # _, _, vh = np.linalg.svd(centered)  # SVD axes are arbitrary
    # corner_uv = np.stack([centered @ vh[0], centered @ vh[1]], axis=1)
    corner_uv = np.stack([centered @ u, centered @ v], axis=1)
    polygon = Path(corner_uv)

    return corner_uv, np.stack([u, v, normal]), polygon, centroid


def crop_PCD(
    pcd, corners, vh, polygon, centroid, eps=0.02, filter_points=False, **kwargs
):
    u = vh[0]  # X
    v = vh[1]  # Y
    normal = vh[2]  # smallest singular vector

    # Project pcd.points to (u, v) plane
    def project_uv_plane(pts):
        rel_points = pts - centroid
        proj_u = rel_points @ u
        proj_v = rel_points @ v
        proj_2d = np.vstack((proj_u, proj_v)).T
        return rel_points, proj_2d

    points = np.asarray(pcd.points)
    rel_points, proj_2d = project_uv_plane(points)

    # Check inclusion
    mask = polygon.contains_points(proj_2d)
    point_plane_dist = np.abs(rel_points @ normal)
    final_mask = mask & (point_plane_dist < eps)
    cropped_pcd = pcd.select_by_index(np.where(final_mask)[0])

    if filter_points:
        cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.005)
        cropped_pcd, _ = cropped_pcd.remove_statistical_outlier(
            nb_neighbors=50, std_ratio=0.05
        )
        cropped_pcd, _ = cropped_pcd.remove_radius_outlier(nb_points=30, radius=0.05)

    if kwargs.get("show_PCD", False):
        o3d.visualization.draw_geometries([cropped_pcd])
        print("Press 'q' to Exit.")

    if kwargs.get("ply_path", False):
        o3d.io.write_point_cloud(kwargs["ply_path"], cropped_pcd)
        print(f"Saved point cloud to: {kwargs['ply_path']}")

    cropped_points = np.asarray(cropped_pcd.points)
    _, flattened_points = project_uv_plane(cropped_points)

    return flattened_points


def flatten(A, cropped_points, DOI_size, image_size=224, buffer=10):
    points = transform_pts(cropped_points, A)

    points = points[:, :2] * image_size / DOI_size
    mask = (
        (points[:, 0] >= buffer)
        & (points[:, 0] < image_size - buffer)
        & (points[:, 1] >= buffer)
        & (points[:, 1] < image_size - buffer)
    )
    points = points[mask]

    return points.astype(int)
