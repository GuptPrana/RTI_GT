import numpy as np
import open3d as o3d
from matplotlib.path import Path

"""
Select the corners of DOI for picked_points
in same order as dst_points below.
"""


def affine_matrix(picked_points, dst_pts, lstsq=True):
    N = picked_points.shape[0]
    assert dst_pts.shape[0] == N
    src_h = np.hstack([picked_points, np.ones((N, 1))])
    dst_h = np.hstack([dst_pts, np.ones((N, 1))])

    if lstsq:
        A_T, residuals, _, _ = np.linalg.lstsq(src_h, dst_h, rcond=None)
        return A_T.T
    
    A = np.linalg.solve(src_h.T, dst_h.T).T
    return A


def transform_pts(points, A):
    points_ = np.hstack([points, np.ones((points.shape[0], 1))])
    print(A.shape)
    print([points_.shape])
    transformed_ = (A @ points_.T).T
    n = A.shape[0] - 1
    transformed_points = transformed_[:, :n] / transformed_[:, [n]]  # scale
    return transformed_points


def define_plane(picked_points):
    # picked_points = np.array([Nx3]), Corners of DOI Plane.
    centroid = picked_points.mean(axis=0)
    centered = picked_points - centroid
    _, _, vh = np.linalg.svd(centered)

    corner_uv = np.stack([centered @ vh[0], centered @ vh[1]], axis=1)
    polygon = Path(corner_uv)
    return corner_uv, vh, polygon, centroid


def crop_PCD(pcd, vh, polygon, centroid, eps=0.01, filter_points=True, **kwargs):
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
        # cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.02)
        cropped_pcd, _ = cropped_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )

    if kwargs.get('show_PCD', False):
        o3d.visualization.draw_geometries([cropped_pcd])
        print("Press 'q' to Exit.")

    if kwargs.get('ply_path', False):
        o3d.io.write_point_cloud(kwargs['ply_path'], cropped_pcd)
        print(f"Saved point cloud to: {kwargs['ply_path']}")
    
    cropped_points = np.asarray(cropped_pcd.points)
    _, flattened_points = project_uv_plane(cropped_points)

    return flattened_points


def flatten(A, cropped_points, DOI_size, image_size=224, buffer=20):
    points = transform_pts(cropped_points, A)

    points = points[:, :2] * image_size / DOI_size
    mask = (
        (points[:, 0] >= buffer)
        & (points[:, 0] < image_size - buffer)
        & (points[:, 1] >= buffer)
        & (points[:, 1] < image_size - buffer)
    )
    points = points[mask]
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], marker='o')
    plt.axis('equal')
    plt.show()

    return points.astype(int)
