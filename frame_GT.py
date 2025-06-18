import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN


def segment(points, plot=False):
    # points = [v1, v2, v3, v4]
    model = DBSCAN(eps=10, min_samples=5).fit(points)
    labels = model.labels_
    clusters = np.unique(labels[labels != -1])
    segmented_points = [points[labels == cluster] for cluster in clusters]

    if plot:
        plt.figure(figsize=(10, 10))
        clusters = set(labels)
        colors = ["red", "blue", "green", "orange"]

        for label in clusters:
            if label == -1:
                color = "gray"
            else:
                color = colors[label % len(colors)]
            cluster_mask = labels == label
            plt.scatter(
                points[cluster_mask, 0],
                points[cluster_mask, 1],
                c=color,
                s=20,
                label=f"{label}",
            )

        plt.title("DBSCAN")
        plt.xlim(0, 224)
        plt.ylim(0, 224)
        plt.legend()
        plt.grid(True)
        plt.show()

    return segmented_points


def make_gt(segmented_points, frame_size=224, plot=False):
    gt = np.zeros((frame_size, frame_size), dtype=np.uint8)

    for item in segmented_points:
        hull = ConvexHull(item)
        hull_points = item[hull.vertices]
        shape = Polygon(hull_points)
        mask = np.array(shape.exterior.coords).round().astype(np.int32)
        mask = mask.reshape((-1, 1, 2))
        cv2.fillPoly(gt, [mask], 1)

    if plot:
        plt.imshow(gt, cmap="gray", origin="lower")

    return gt


def motion_compensated_gt(gt_frames):
    pass
