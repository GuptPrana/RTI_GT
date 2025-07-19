import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.mixture import GaussianMixture


def segment(points, object_count=2, plot=False):
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


def object_shape(points):
    # try alpha shape (convex hull may be distorted due to outlier points)
    # or smoothen in preprocessing
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    return Polygon(hull_points)


def make_gt(segmented_points, image_size=224, plot=False):
    gt = np.zeros((image_size, image_size), dtype=np.uint8)
    for points in segmented_points:
        shape = object_shape(points)
        mask = np.array(shape.exterior.coords).round().astype(np.int32)
        mask = mask.reshape((-1, 1, 2))
        cv2.fillPoly(gt, [mask], 1)

    if plot:
        plt.imshow(gt, cmap="gray", origin="lower")
        plt.show()

    return gt


def motion_compensated_gt(gt_frames):
    pass  # return np.logical_or.reduce(gt_frames)
