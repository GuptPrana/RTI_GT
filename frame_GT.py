import alphashape
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from sklearn.mixture import GaussianMixture


def segment(points, object_count, plot):
    gmm = GaussianMixture(n_components=object_count)
    # Can consider RGB as additional features
    gmm.fit(points)
    labels = gmm.predict(points)
    clusters = np.unique(labels)
    segmented_points = [points[labels == cluster] for cluster in clusters]

    if plot:
        plt.figure(figsize=(10, 10))
        colors = ["red", "blue", "green", "orange"]

        for cluster in clusters:
            mask = labels == cluster
            color = colors[cluster % len(colors)]
            plt.scatter(
                points[mask, 0],
                points[mask, 1],
                c=color,
                s=20,
                label=f"{cluster}",
            )

        plt.title("GMM Clustering")
        plt.xlim(0, 224)
        plt.ylim(0, 224)
        plt.legend()
        plt.grid(True)
        plt.show()

    return segmented_points


def object_shape(points, **kwargs):
    hull = ConvexHull(points, **kwargs)
    hull_points = hull.points[hull.vertices]
    return hull_points


def alpha_shape(points, alpha):
    # try alpha shape (convex hull may be distorted due to outlier points)
    hull_points = object_shape(points)
    hull_tree = cKDTree(hull_points)

    alpha_low = alpha["alpha_low"]
    alpha_high = alpha["alpha_high"]
    eps = alpha["eps"]

    dists, _ = hull_tree.query(points)
    dense_points = points[dists < eps]
    sparse_points = points[dists >= eps]

    shapes = []
    if len(dense_points) >= 4:
        dense_shape = alphashape.alphashape(dense_points, alpha_low)
        if dense_shape:
            shapes.append(dense_shape)

    if len(sparse_points) >= 4:
        sparse_shape = alphashape.alphashape(sparse_points, alpha_high)
        if sparse_shape:
            shapes.append(sparse_shape)

    if not shapes:
        return Polygon(hull_points)  # Fallback

    combined = unary_union(shapes)

    if isinstance(combined, MultiPolygon):
        all_points = np.concatenate(
            [np.array(p.exterior.coords) for p in combined.geoms]
        )
        combined = Polygon(object_shape(all_points))

    return combined


def make_gt(segmented_points, image_size, alpha, plot):
    gt = np.zeros((image_size, image_size), dtype=np.uint8)
    for points in segmented_points:
        if alpha:
            shape = alpha_shape(points, alpha)
        else:
            shape = Polygon(object_shape(points))
        mask = np.array(shape.exterior.coords).round().astype(np.int32)
        mask = mask.reshape((-1, 1, 2))
        cv2.fillPoly(gt, [mask], 1)

    if plot:
        plt.imshow(gt, cmap="gray", origin="lower")
        plt.show()

    return gt


def motion_compensated_gt(gt_frames):
    pass  # return np.logical_or.reduce(gt_frames)
