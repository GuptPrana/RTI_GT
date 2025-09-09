import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box

from masks.frame_GT import *


def angle_from_camera(point, camera_pos):
    vec = point - camera_pos
    return np.arctan2(vec[1], vec[0]) % (2 * np.pi)


def rotate(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def extend_ray(point, camera_pos, image_size, eps=0):
    frame_box = box(0, 0, image_size, image_size)
    direction = (point - camera_pos).astype(np.float64)
    direction /= np.linalg.norm(direction)

    if eps:
        direction = rotate(direction, eps)
    ray = LineString([point, point + direction * 1000])
    inter = ray.intersection(frame_box.boundary)

    if not isinstance(inter, Point):
        # TODO: Add logger
        return None
    return np.array([inter.x, inter.y])


def centroid(points):
    return np.mean(points, axis=0)


def dist(p1, p2):
    return np.sum(np.abs(p2 - p1))


def make_shadow(points, camera_pos, image_size, eps=0):
    frame_corners = np.array(
        [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]
    )

    hull_points = object_shape(points, qhull_options="QJ")
    # angles for hull_points relative to camera_pos
    angles = np.array([angle_from_camera(p, camera_pos) for p in hull_points])
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_hull = hull_points[sorted_indices]

    # pick smaller arc using first difference
    diffs = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2 * np.pi])
    split_idx = np.argmax(diffs)
    arc_angles = np.roll(sorted_angles, -(split_idx + 1))
    arc_hull = np.roll(sorted_hull, -(split_idx + 1), axis=0)

    p_left, p_right = arc_hull[0], arc_hull[-1]
    angle_left, angle_right = arc_angles[0], arc_angles[-1]

    # Widen shadow with eps
    inter_left = extend_ray(p_left, camera_pos, image_size, -eps)
    inter_right = extend_ray(p_right, camera_pos, image_size, eps)
    corner_angles = np.array([angle_from_camera(p, camera_pos) for p in frame_corners])

    # Handle 2*np.pi wraparound
    if angle_left > angle_right:
        in_between_corners = frame_corners[
            (corner_angles >= angle_left) | (corner_angles <= angle_right)
        ]
    else:
        in_between_corners = frame_corners[
            (corner_angles >= angle_left) & (corner_angles <= angle_right)
        ]

    # Occlusion Polygon
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

    # TODO: Add Logger
    occlusion_polygon = Polygon(all_occlusion_points)
    if not occlusion_polygon.is_valid:
        print(f"Erroneous Polygon Area:", occlusion_polygon.area)
        occlusion_polygon = occlusion_polygon.buffer(0)

    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    coords = (
        np.array(occlusion_polygon.exterior.coords)
        .round()
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )
    cv2.fillPoly(mask, [coords], 1)
    return mask


def filter_points(points, view_masks):
    mask = np.logical_and.reduce(view_masks)
    points = points.round().astype(int)
    points = np.clip(points, 0, mask.shape(0) - 1)
    keep = mask[points[:, 1], points[:, 0]] == 0
    return points[keep]


def make_buffer_mask(image_size, buffer, corner_mult=3):
    cmask = np.zeros((image_size, image_size), dtype=np.uint8)
    cmask[:buffer, :] = 1
    cmask[-buffer:, :] = 1
    cmask[:, :buffer] = 1
    cmask[:, -buffer:] = 1
    if not corner_mult:
        return cmask

    y, x = np.ogrid[:image_size, :image_size]
    buffer = buffer * corner_mult
    tl = x + y < buffer
    tr = (image_size - 1 - x) + y < buffer
    bl = x + (image_size - 1 - y) < buffer
    br = (image_size - 1 - x) + (image_size - 1 - y) < buffer
    cmask[tl | tr | bl | br] = 1

    return cmask


def make_final_cmask(
    all_points,
    cameras,
    buffer_mask,
    object_count=2,
    image_size=112,
    alpha=None,
    plot=True,
    eps=0.0,
):
    masks = []
    keep_points = []
    for view in range(len(all_points)):
        points = segment_GMM(all_points[view], object_count, plot, image_size)

        objects = []
        for object_id in range(len(points)):
            objects.append(
                (object_id, dist(centroid(points[object_id]), cameras[view]))
            )
        ordered = sorted(objects, key=lambda x: x[1])

        view_masks = []
        for cluster in ordered:
            if len(view_masks):
                # filter out occluded/smudged points before GT
                keep_points.append(filter_points(points[cluster[0]], view_masks))
            shadow_mask = make_shadow(cluster, cameras[view], image_size, eps)
            view_masks.append(shadow_mask)
        masks.append(np.logical_and.reduce(view_masks))

    gt = make_gt(np.vstack(keep_points), object_count, image_size, alpha, plot)

    # masks = [cmask1, ..., cmask4]
    cmask = np.logical_and.reduce(masks)
    # ignore pixels inside gt shapes
    cmask[gt == 1] = 0.0
    # buffer frame edges
    cmask = np.logical_or.reduce([cmask, buffer_mask])
    cmask = np.logical_not(cmask)

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(cmask, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
        plt.title("Uncertainty Mask (0: Unoccupied, 1: Occluded/Occupied)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # TODO: Check for failed GT and Logger

    return gt, cmask
