import alphashape
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from config import GT_Config


def spatial_filter(points):
    nbrs = NearestNeighbors(radius=GT_Config.spatial_filter_rad).fit(points)
    neigh_counts = np.array(
        [len(ind) for ind in nbrs.radius_neighbors(points, return_distance=False)]
    )
    return points[neigh_counts >= GT_Config.spatial_filter_neigh]


def segment_GMM(points, object_count, plot, image_size):
    # points = spatial_filter(points)
    gmm = GaussianMixture(n_components=object_count)
    gmm.fit(points)
    # labels = gmm.predict(points)

    probs = gmm.predict_proba(points)
    # Max confidence for each point and the corresponding label
    max_probs = np.max(probs, axis=1)
    labels = np.argmax(probs, axis=1)

    confidence_thresh = GT_Config.GMM_thresh
    keep_indices = max_probs >= confidence_thresh
    points = points[keep_indices]
    labels = labels[keep_indices]

    clusters = np.unique(labels)
    segmented_points = [points[labels == cluster] for cluster in clusters]

    for object_id in range(len(segmented_points)):
        segmented_points[object_id] = spatial_filter(segmented_points[object_id])

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        colors = ["red", "blue", "green", "orange"]

        for cluster in clusters:
            mask = labels == cluster
            color = colors[cluster % len(colors)]
            axes[0].scatter(
                points[mask, 0],
                points[mask, 1],
                c=color,
                s=20,
                label=f"Object {cluster}",
            )
        axes[0].set_title("GMM Clustering")
        axes[0].set_xlim(0, image_size)
        axes[0].set_ylim(0, image_size)
        axes[0].legend()
        axes[0].grid(True)

        for idx, seg in enumerate(segmented_points):
            color = colors[idx % len(colors)]
            axes[1].scatter(
                seg[:, 0],
                seg[:, 1],
                c=color,
                s=20,
                label=f"Object {idx}",
            )
        axes[1].set_title("Segmented Points")
        axes[1].set_xlim(0, image_size)
        axes[1].set_ylim(0, image_size)
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    return segmented_points


def smooth_hull(shape, tolerance=1.0):
    points = np.array(shape.exterior.coords)
    hull = MultiPoint(points).convex_hull
    simplified = hull.simplify(tolerance, preserve_topology=True)
    return simplified


def object_shape(points, **kwargs):
    hull = ConvexHull(points, **kwargs)
    hull_points = hull.points[hull.vertices]
    return hull_points


def alpha_shape(points, alpha):
    shape = alphashape.alphashape(points, alpha)
    if isinstance(shape, MultiPolygon):
        all_points = np.concatenate([np.array(p.exterior.coords) for p in shape.geoms])
        # shape = Polygon(object_shape(all_points))
        shape = alphashape.alphashape(all_points, 0.05)
    # shape = smooth_hull(shape)
    return shape


def reduce_gt(masks):
    dilated_masks = []
    for mask in masks:
        # check for touching objects
        dilated_masks.append(binary_dilation(mask, structure=np.ones((3, 3))))

    if not np.any(np.logical_and(*dilated_masks)):
        return np.logical_or(*masks)
    else:
        while np.any(np.logical_and(*dilated_masks)):
            for m in range(len(dilated_masks)):
                dilated_masks[m] = binary_erosion(
                    dilated_masks[m], structure=np.ones((5, 5))
                )
        return np.logical_or(*dilated_masks)


def make_gt(keep_points, object_count, image_size, plot, minpoints=GT_Config.minpoints):
    segmented_points = segment_GMM(keep_points, object_count, plot, image_size)

    masks = []
    for points in segmented_points:
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        if len(points) < minpoints:
            return None

        if GT_Config.alpha:
            try:
                shape = alpha_shape(points, GT_Config.alpha)
                shape_mask = np.array(shape.exterior.coords).round().astype(np.int32)
            except Exception as e:
                shape = Polygon(object_shape(points))
                shape_mask = np.array(shape.exterior.coords).round().astype(np.int32)
                print(e)
        else:
            shape = Polygon(object_shape(points))
            shape_mask = np.array(shape.exterior.coords).round()
            # if not np.array_equal(shape_mask[0], shape_mask[-1]):
            #     shape_mask = np.vstack([shape_mask, shape_mask[0]])
            # plt.imshow(mask, cmap="gray", origin="lower")
            # plt.plot(shape_mask[:,0], shape_mask[:,1], "r-")
            # plt.show()

        shape_mask = shape_mask.reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [shape_mask], 1)
        masks.append(mask)

    gt = reduce_gt(masks)

    if plot:
        plt.imshow(gt, cmap="gray", origin="lower")
        plt.show()

    return gt.astype(np.uint8)


def motion_compensated_gt(gt_frames):
    pass  # return np.logical_or.reduce(gt_frames)
