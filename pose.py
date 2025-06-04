import bisect
import os

import cv2
import numpy as np
from ultralytics import YOLO

# import torch
# import matplotlib.pyplot as plt


def extract_timestamp(filename):
    return int(filename.split(".")[0])


def apply_homography(x, y, H):
    # can access H by saved np.array
    point = np.array([x, y, 1])
    tr = H @ point
    tr /= tr[2]
    return int(tr[0]), int(tr[1])


def getCoordinates(camera, transform):
    image_paths = os.listdir(camera)
    model2 = YOLO("yolov8x-pose-p6.pt")
    data = []
    for path in image_paths:
        image = cv2.imread(os.path.join(camera, path))
        results = model2(image)
        if len(results[0].keypoints.xy) == 0:
            continue  # No keypoints
        # For visualization
        # img = results[0].plot()  # This plots the detections on the image
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(path + '_pose', img_rgb)
        # cv2.imshow(img_rgb)
        # cv2.waitKey(0)
        # cv2.destroykeypoints.xy[0]
        coords = []
        coordsT = []
        for person in range(len(results[0].keypoints.xy)):
            keypoints = results[0].keypoints.xy[person]
            ### Should check if 15 and 16 are present.
            left_foot = keypoints[15].tolist()  # left foot
            right_foot = keypoints[16].tolist()  # right foot
            x = (left_foot[0] + right_foot[0]) / 2
            y = (left_foot[1] + right_foot[1]) / 2
            coords.append([x, y])
            coordsT.append([apply_homography(x, y, transform)])

        data.append([extract_timestamp(path), coords, coordsT])
    return data


def synchronize(dfs, eps):
    data = [dfs[0]]
    for col in range(1, dfs.shape[0] + 1):
        ### "left join"
        ### All col must be sorted
        colA = dfs[0][:, 0]
        colB = dfs[col][:, 0]

        indices = []
        for entry in colA:
            idx = bisect.bisect_left(colB, entry)

            # nearest = []
            # if idx > 0:
            #     diff = abs(entry - colB[idx - 1])
            #     if diff < eps:  nearest.append((diff, colB[idx - 1]))
            # if idx < len(colB):
            #     diff = abs(entry - colB[idx])
            #     if diff < eps:  nearest.append((diff, colB[idx]))

            # if nearest:
            #     diff, best_match = min(nearest, key=lambda x: x[0])

            diffs = []
            if idx > 0:
                diffs.append(abs(entry - colB[idx - 1]))
            if idx < len(colB):
                diffs.append(abs(entry - colB[idx]))
            if diffs[0] < diffs[1]:
                index = idx - 1
            else:
                index = idx
            indices.append(index)

        data.append(dfs[col][indices])


def main():
    camerapaths = ["realsense_data/cam1/", "realsense_data/cam2/"]
    dfs = []
    for view in range(len(camerapaths)):
        transform = np.load(f"realsense_data/transform_{view}.npy")
        dfs.append(getCoordinates(camerapaths[view], transform))

    dfs = np.array(dfs)
    data = synchronize(dfs)
    np.save("coordinates.npy", data)


if __name__ == "__main__":
    main()
