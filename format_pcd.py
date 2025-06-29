import numpy as np
import open3d as o3d
from matplotlib.path import Path

"""
Can select the corners of DOI for picked_points
in same order as dst_points below.
"""

dst_pts = np.array([[0, 0, 1.5], [3, 0, 1.5], [0, 3, 1.5], [3, 3, 1.5]])


def affine_matrix(picked_points, dst_pts=dst_pts):
    N = picked_points.shape[0]
    assert dst_pts.shape[0] == N
    src_h = np.hstack([picked_points, np.ones((N, 1))])
    dst_h = np.hstack([dst_pts, np.ones((N, 1))])

    # dst_h = src_h @ A.T --> A.T = pinv(src_h) @ dst_h
    A_T, residuals, _, _ = np.linalg.lstsq(src_h, dst_h, rcond=None)
    return A_T.T


def transform_pcd(points, A, as_pcd=False):
    points_ = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_ = (A @ points_.T).T
    transformed_points = transformed_[:, :3] / transformed_[:, 3:]  # scale

    if as_pcd:
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        return transformed_pcd

    return transformed_points


def define_plane(picked_points):
    # picked_points = np.array([Nx3]), Corners of DOI Plane.
    centroid = picked_points.mean(axis=0)
    centered = picked_points - centroid
    _, _, vh = np.linalg.svd(centered)

    corner_uv = np.stack([centered @ vh[0], centered @ vh[1]], axis=1)
    polygon = Path(corner_uv)
    return vh, polygon, centroid


def crop_PCD(pcd, vh, polygon, centroid, eps=0.01, show_PCD=False):
    u = vh[0]  # X
    v = vh[1]  # Y
    # Normal = smallest singular vector
    normal = vh[2]

    # Project pcd.points to (u, v) plane
    points = np.asarray(pcd.points)
    rel_points = points - centroid
    proj_u = rel_points @ u
    proj_v = rel_points @ v
    proj_2d = np.vstack((proj_u, proj_v)).T

    # Check inclusion
    mask = polygon.contains_points(proj_2d)
    point_plane_dist = np.abs(rel_points @ normal)
    final_mask = mask & (point_plane_dist < eps)
    cropped_pcd = pcd.select_by_index(np.where(final_mask)[0])

    if show_PCD:
        o3d.visualization.draw_geometries([cropped_pcd])
        print("Press 'q' to Exit.")

    return cropped_pcd.points


def flatten(A, cropped_points, image_size=224):
    points = transform_pcd(cropped_points, A)
    points = points[:, :2] * image_size / dst_pts.max()

    # if not np.all((points >= 0) & (points <= image_size)):
    #     raise ValueError("Out of range values!")
    return points.astype(int)
