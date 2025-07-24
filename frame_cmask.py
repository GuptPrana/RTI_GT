import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from frame_GT import *


def make_cmask(camera_pos, segmented_viewpts, image_size=224, plot=False):
    frame_box = box(0, 0, image_size, image_size)
    frame_corners = np.array(
        [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]
    )

    def angle_from_camera(pt):
        vec = pt - camera_pos
        return np.arctan2(vec[1], vec[0])  # % (2 * np.pi)

    def extend_ray(point):
        direction = (point - camera_pos).astype(np.float64)
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

    shadow_polygons = []
    object_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for object_points in segmented_viewpts:  # object-wise shadow maps
        # skip if no object_points in given view
        if len(object_points) < 3:
            continue

        hull = ConvexHull(object_points, qhull_options="QJ")
        hull_points = object_points[hull.vertices]

        hull_polygon = Polygon(hull_points)
        if hull_polygon.is_valid and not hull_polygon.is_empty:
            coords = (
                np.array(hull_polygon.exterior.coords)
                .round()
                .astype(np.int32)
                .reshape((-1, 1, 2))
            )
            cv2.fillPoly(object_mask, [coords], 1)

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

        occlusion_polygon = Polygon(all_occlusion_points)
        if not occlusion_polygon.is_valid:
            print(f"Erroneous Polygon Area:", occlusion_polygon.area)
            occlusion_polygon = occlusion_polygon.buffer(0)
        if occlusion_polygon.is_valid:
            shadow_polygons.append(occlusion_polygon)

        # plt.figure()
        # plt.scatter(*all_occlusion_points.T, marker='o')
        # plt.title("Polygon Candidate")
        # plt.scatter(*camera_pos, color='red', label='Camera')
        # plt.xlim((0, 224))
        # plt.ylim((0, 224))
        # plt.legend()
        # plt.show()

    mask = np.zeros((image_size, image_size), dtype=np.float32)
    try:
        final_occlusion = unary_union(shadow_polygons)

        if isinstance(final_occlusion, Polygon):
            polys = [final_occlusion]
        elif isinstance(final_occlusion, MultiPolygon):
            polys = list(final_occlusion.geoms)
        else:
            raise TypeError(f"Unexpected geometry type: {type(final_occlusion)}")

        for poly in polys:
            coords = (
                np.array(poly.exterior.coords)
                .round()
                .astype(np.int32)
                .reshape((-1, 1, 2))
            )
            cv2.fillPoly(mask, [coords], 1.0)
        mask[object_mask == 1] = 0
    except Exception as e:
        print("Mask creation failed: {e}")

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(mask, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        plt.title("Uncertainty Mask (0: Unoccupied, 1: Occluded/Occupied)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return mask


def make_final_cmask(
    all_points, cameras, object_count, image_size=224, cmask=True, plot=True
):
    segmented_points = segment(np.vstack(all_points), object_count)

    gt = make_gt(segmented_points, image_size=image_size, plot=plot)
    if not cmask:
        return gt, None

    # object_wise x view_wise
    def row_mask(a, b):
        a_view = a.view([("", a.dtype)] * a.shape[1])
        b_view = b.view([("", b.dtype)] * b.shape[1])
        return np.isin(a_view, b_view).reshape(-1)

    masks = []
    # Make Uncertainty Masks
    for view in range(len(cameras)):
        segmented_viewpts = [
            all_points[view][row_mask(all_points[view], object_points)]
            for object_points in segmented_points
        ]
        mask = make_cmask(cameras[view], segmented_viewpts, image_size=image_size)
        masks.append(mask)

    # masks = [cmask1, ..., cmask4]
    cmask = np.logical_and.reduce(masks)
    # ignore pixels inside gt shapes
    cmask[gt == 1] = 0.0

    if plot:
        plt.imshow(cmask, cmap="gray", origin="lower")
        plt.title("Uncertainty Map")
        plt.show()

    return gt, cmask
