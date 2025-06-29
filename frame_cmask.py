import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from frame_GT import *


def make_cmask(camera_pos, points, frame_size=224, plot=False):
    frame_box = box(0, 0, frame_size, frame_size)
    frame_corners = np.array(
        [[0, 0], [0, frame_size], [frame_size, frame_size], [frame_size, 0]]
    )

    def angle_from_camera(pt):
        vec = pt - camera_pos
        return np.arctan2(vec[1], vec[0])  # % (2 * np.pi)

    def extend_ray(point):
        direction = point - camera_pos
        direction /= np.linalg.norm(direction)
        ray = LineString([point, point + direction * 1000])
        inter = ray.intersection(frame_box.boundary)
        if inter.is_empty:
            return None
        if isinstance(inter, Point):
            return np.array([inter.x, inter.y])
        else:
            points = [p for p in inter]
            dists = [np.linalg.norm(np.array([p.x, p.y]) - point) for p in points]
            farthest = points[np.argmax(dists)]
            return np.array([farthest.x, farthest.y])

    segmented_points = segment(points)
    shadow_polygons = []

    for n in range(len(segmented_points)):
        hull = ConvexHull(segmented_points[n])
        hull_points = segmented_points[n][hull.vertices]

        # angles for hull_points relative to camera_pos
        angles = np.array([angle_from_camera(p) for p in hull_points])
        sorted_indices = np.argsort(angles)
        sorted_hull = hull_points[sorted_indices]

        p_left = sorted_hull[0]
        p_right = sorted_hull[-1]
        inter_left = extend_ray(p_left)
        inter_right = extend_ray(p_right)

        corner_angles = np.array([angle_from_camera(p) for p in frame_corners])
        angle_left = angle_from_camera(p_left)
        angle_right = angle_from_camera(p_right)

        if angle_left > angle_right:
            angle_left, angle_right = angle_right, angle_left

        if angle_left > angle_right:
            in_between_corners = frame_corners[
                (corner_angles >= angle_left) | (corner_angles <= angle_right)
            ]
        else:
            in_between_corners = frame_corners[
                (corner_angles >= angle_left) & (corner_angles <= angle_right)
            ]

        # occlusion polygon
        all_occlusion_points = np.vstack(
            [
                p_left,
                *sorted_hull[
                    (angles[sorted_indices] >= angle_left)
                    & (angles[sorted_indices] <= angle_right)
                ],
                p_right,
                inter_right,
                *in_between_corners,
                inter_left,
            ]
        )

        # angles = np.arctan2(all_occlusion_points[:,1] - camera_pos[1], all_occlusion_points[:,0] - camera_pos[0])
        # sort_idx = np.argsort(angles)
        # ordered_occlusion_points = all_occlusion_points[sort_idx]
        # occlusion_polygon = Polygon(ordered_occlusion_points)
        raw_occlusion_polygon = Polygon(all_occlusion_points)
        shadow_polygons.append(raw_occlusion_polygon)

    final_occlusion = unary_union(shadow_polygons)

    if isinstance(final_occlusion, Polygon):
        polys = [final_occlusion]
    elif isinstance(final_occlusion, MultiPolygon):
        polys = list(final_occlusion.geoms)
    else:
        raise TypeError("Expected Polygon or MultiPolygon")

    mask = np.zeros((frame_size, frame_size), dtype=np.float32)
    for poly in polys:
        coords = (
            np.array(poly.exterior.coords).round().astype(np.int32).reshape((-1, 1, 2))
        )
        cv2.fillPoly(mask, [coords], 1.0)

    if plot:
        # Plotting both masks
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(mask, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        axs[0].set_title(
            "Uncertainty Mask (0: Object/Unoccupied, 1: Occluded from All Views)"
        )
        axs[0].axis("off")

        # convex_hull = make_gt(segmented_points)
        # axs[1].imshow(convex_hull, cmap='gray', origin='lower')
        # axs[1].set_title('Binary Mask (1: object, 0: background)')
        # axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return mask


def make_final_gt(points):
    segmented_points = segment(points)
    gt = make_gt(segmented_points)
    return gt


def make_final_cmask(points, cameras):
    # Make GT
    gt = make_final_gt(points)
    masks = []
    # Make Uncertainty Masks
    for view_id in range(len(cameras)):
        masks.append(make_cmask(cameras[view_id], points[view_id]))

    # masks = [cmask1, ..., cmask4]
    cmask = np.logical_and.reduce(masks)
    # ignore pixels inside gt shapes
    cmask[gt == 1] = 0.0

    return gt, cmask
