import bisect
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
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


def get_sync_timestamps(camerapaths, filetype=".jpg", eps=500, filter=True, save=True):
    timestamps = []
    for path in camerapaths:
        timestamps.append(
            [
                int(f.replace(filetype, ""))
                for f in os.listdir(path)
                if f.endswith(filetype)
            ]
        )

    for view in range(1, len(camerapaths)):
        ### "left join"
        colA = timestamps[0]
        colB = timestamps[view]

        sync = []
        for entry in colA:
            idx = bisect.bisect_left(colB, entry)

            nearest = []
            if idx > 0:
                diff = abs(entry - colB[idx - 1])
                if diff < eps:
                    nearest.append((diff, colB[idx - 1]))
            if idx < len(colB):
                diff = abs(entry - colB[idx])
                if diff < eps:
                    nearest.append((diff, colB[idx]))

            if nearest:
                diff, best_match = min(nearest, key=lambda x: x[0])
            else:
                best_match = np.nan  # frame will be deleted
            sync.append(best_match)

            # diffs = []
            # if idx > 0:
            #     diffs.append(abs(entry - colB[idx - 1]))
            # if idx < len(colB):
            #     diffs.append(abs(entry - colB[idx]))
            # if diffs[0] < diffs[1]:
            #     index = idx - 1
            # else:
            #     index = idx
            # indices.append(index)

        timestamps[view] = sync
    timestamps = np.array(timestamps)  # len(col{k}) = len(colA)

    if filter:
        keep = ~np.any(np.isnan(timestamps), axis=0)
        timestamps = timestamps[:, keep]

    if save:
        np.save("timestamps.npy", timestamps)

    return timestamps


def get_aggregate(allview_coordsT):
    # Hungarian Algorithm
    people_by_view = [np.array(view) for view in allview_coordsT]

    # Use first view as reference
    ref_coords = people_by_view[0]
    matched_coords = [ref_coords]

    for other_coords in people_by_view[1:]:
        if len(other_coords) == 0:
            matched_coords.append(np.full_like(ref_coords, np.nan))
            continue

        cost_matrix = np.linalg.norm(
            ref_coords[:, np.newaxis, :] - other_coords[np.newaxis, :, :], axis=2
        )  # Shape: (n_ref, n_other)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = np.full_like(ref_coords, np.nan)
        matched[row_ind] = other_coords[col_ind]
        matched_coords.append(matched)

    # Stack and average
    stacked = np.stack(matched_coords)  # Shape: (views, n_persons, 2)
    avg_coords = np.nanmean(stacked, axis=0)
    return avg_coords.tolist()


def load_transforms(views):
    transforms = []
    for view in range(1, views + 1):
        transform = np.load(f"realsense_data/transform_{view}.npy")
        transforms.append(transform)
    return np.array(transforms)


def get_coordinates(model, camerapaths, timestamp, transforms):
    allview_coords = []
    allview_coordsT = []
    for view in timestamp:
        path = os.path.join(camerapaths[view], str(timestamp[view]) + ".jpg")
        image = cv2.imread(path)
        results = model(image)
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
            coordsT.append([apply_homography(x, y, transforms[view])])

        allview_coords.append(coords)
        allview_coordsT.append(coordsT)

    return [
        timestamp[0],
        allview_coords,
        allview_coordsT,
        get_aggregate(allview_coordsT),
    ]


def multiview_coordinates(camerapaths, timestamps, transforms, save=True):
    model = YOLO("yolov8x-pose-p6.pt")
    all_timestamps = []
    for time in range(timestamps.shape[1]):
        result = get_coordinates(model, camerapaths, timestamps[:, time], transforms)
        all_timestamps.append(result)

    all_timestamps = np.array(all_timestamps, dtype=object)

    if save:
        np.save("all_timestamps_complete.npy", all_timestamps)
        np.save("all_coordinates.npy, ", all_timestamps[:, [0, 3]])

    return


def main():
    camerapaths = ["realsense_data/cam1/", "realsense_data/cam2/"]
    timestamps = get_sync_timestamps(camerapaths)
    transforms = load_transforms(len(timestamps))
    multiview_coordinates(camerapaths, timestamps, transforms)


if __name__ == "__main__":
    main()
