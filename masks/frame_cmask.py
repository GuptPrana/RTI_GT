import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
from shapely.geometry import LineString, Point, Polygon, box

from config import GT_Config
from masks.frame_GT import *
from masks.helpers import logger


def angle_from_camera(point, camera_pos):
    vec = point - camera_pos
    return np.arctan2(vec[1], vec[0]) % (2 * np.pi)


def rotate(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def cleanup(inter, camera_pos):
    if inter.is_empty:
        return None
    if isinstance(inter, Point):
        return np.array([inter.x, inter.y])
    else:
        intersects = list(inter.geoms)
        dists = [dist(np.array([p.x, p.y]), camera_pos) for p in intersects]
        farthest = intersects[np.argmax(dists)]
        return np.array([farthest.x, farthest.y])


def extend_rays(p_left, p_right, camera_pos, image_size, eps):
    frame_box = box(0, 0, image_size, image_size)

    dir_left = (p_left - camera_pos).astype(np.float64)
    dir_left /= np.linalg.norm(dir_left)
    dir_right = (p_right - camera_pos).astype(np.float64)
    dir_right /= np.linalg.norm(dir_right)

    if eps:
        dir_left = rotate(dir_left, -eps)
        dir_right = rotate(dir_right, eps)

    # ray = LineString([point, point + direction * 1000])
    ray_left = LineString([camera_pos, camera_pos + dir_left * 1000])
    ray_right = LineString([camera_pos, camera_pos + dir_right * 1000])

    if eps:
        chord_dir = p_right - p_left
        chord = LineString([p_left - 1000 * chord_dir, p_right + 1000 * chord_dir])
        p_left_n = ray_left.intersection(chord)
        p_right_n = ray_right.intersection(chord)

        if not p_left_n.is_empty:
            p_left = cleanup(p_left_n, camera_pos)
        if not p_right_n.is_empty:
            p_right = cleanup(p_right_n, camera_pos)

    inter_left = cleanup(ray_left.intersection(frame_box.boundary), camera_pos)
    inter_right = cleanup(ray_right.intersection(frame_box.boundary), camera_pos)

    return inter_left, inter_right, p_left, p_right


def centroid(points):
    return np.mean(points, axis=0)


def dist(p1, p2):
    return np.sum(np.abs(p2 - p1))


def dist_1d(pt, camera_pos, image_size):
    # assume all cameras point to center
    DOI_center = [image_size / 2, image_size / 2]
    # scalar projection
    camera_dir = (DOI_center - camera_pos).astype(np.float64)
    camera_dir /= np.linalg.norm(camera_dir)
    pt_vec = (pt - camera_pos).astype(np.float64)
    return np.dot(pt_vec, camera_dir)


def make_shadow(points, camera_pos, image_size, plot):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    frame_corners = np.array(
        [[0, 0], [0, image_size], [image_size, image_size], [image_size, 0]]
    )

    if len(points) < 3:
        return mask
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

    eps = GT_Config.eps
    p_left, p_right = arc_hull[0], arc_hull[-1]
    angle_left = (arc_angles[0] - np.deg2rad(eps)) % (2 * np.pi)
    angle_right = (arc_angles[-1] + np.deg2rad(eps)) % (2 * np.pi)

    # Widen shadow with eps
    inter_left, inter_right, p_left, p_right = extend_rays(
        p_left, p_right, camera_pos, image_size, np.deg2rad(eps)
    )
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

    occlusion_polygon = Polygon(all_occlusion_points)
    if not occlusion_polygon.is_valid or not isinstance(occlusion_polygon, Polygon):
        print(f"Erroneous Polygon Area:", occlusion_polygon.area)
        occlusion_polygon = occlusion_polygon.buffer(0)
        if not occlusion_polygon.is_valid or not isinstance(occlusion_polygon, Polygon):
            print(f"Cannot Fix Polygon Area:", occlusion_polygon.area)
            raise ValueError("Self-intersecting or Open Shadow Polygon")

    coords = (
        np.array(occlusion_polygon.exterior.coords)
        .round()
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )
    cv2.fillPoly(mask, [coords], 1)

    if plot:
        plot_mask(np.flipud(mask))

    return mask


def plot_mask(mask):
    plt.figure(figsize=(5, 5))
    plt.imshow(np.flipud(mask), cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    plt.title("Uncertainty Mask (0: Unoccupied, 1: Occluded/Occupied)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def filter_points(points, mask):
    points = points.round().astype(int)
    points = np.clip(points, 0, mask.shape[0] - 1)
    keep = mask[points[:, 1], points[:, 0]] == 0
    return points[keep]


def make_buffer_mask(image_size):
    cmask = np.zeros((image_size, image_size), dtype=np.uint8)
    buffer = GT_Config.buffer
    corner_mult = GT_Config.corner_mult

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


def touching_buffer(all_points, buffer_mask):
    buffer_mask = binary_dilation(buffer_mask).astype(np.uint8)

    objects_mask = np.zeros_like(buffer_mask)
    hull = object_shape(np.vstack(all_points), qhull_options="QJ")
    coords = (
        np.array(Polygon(hull).buffer(0).exterior.coords)
        .round()
        .astype(np.int32)
        .reshape((-1, 1, 2))
    )
    cv2.fillPoly(objects_mask, [coords], 1)

    return np.logical_and(objects_mask, buffer_mask).sum() > GT_Config.max_overlap


@logger(GT_Config.logfile)
def make_final_cmask(
    all_points,
    cameras,
    buffer_mask,
    filename,
    object_count=2,
    image_size=112,
    plot=True,
):
    if touching_buffer(all_points, buffer_mask):
        # Optional filtering of objects near edge/corner
        raise ValueError("Object(s) along DOI edge/corner")

    # View-wise iterative filtering
    masks = []
    keep_points = []
    for view in range(len(all_points)):
        if len(all_points[view]) < 3:
            raise ValueError(f"Too few global points from a Camera {view}")
        points = segment_GMM(all_points[view], object_count, plot, image_size)

        objects = []
        for object_id in range(len(points)):
            if len(points[object_id]) > 0:
                objects.append(
                    (
                        object_id,
                        dist_1d(centroid(points[object_id]), cameras[view], image_size),
                    )
                )
        ordered = sorted(objects, key=lambda x: x[1])

        view_masks = [np.zeros((image_size, image_size), dtype=np.uint8)]
        for cluster in ordered:
            # filter out occluded/smudged points before GT
            view_mask = np.logical_or.reduce(view_masks)
            keep_points.extend(filter_points(points[cluster[0]], view_mask))
            shadow_mask = make_shadow(
                points[cluster[0]], cameras[view], image_size, plot
            )
            view_masks.append(shadow_mask)
        masks.append(np.logical_or.reduce(view_masks))
        if plot:
            plot_mask(np.flipud(masks[-1]))

    gt = make_gt(np.array(keep_points), object_count, image_size, plot)

    # masks = [cmask1, ..., cmask4]
    cmask = np.logical_and.reduce(masks)
    # ignore pixels inside gt shapes
    cmask[gt == 1] = 0.0
    # buffer frame edges
    cmask = np.logical_or.reduce([cmask, buffer_mask])
    cmask = np.logical_not(cmask)

    if plot:
        plot_mask(np.flipud(cmask))

    return gt, cmask
