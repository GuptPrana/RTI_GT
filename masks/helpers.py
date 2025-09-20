import functools
import json
import os
import traceback

import numpy as np

from pose import linear_timescale


def prepare_intrinsics(intrinsics_paths):
    intrinsics_list = []
    for path in intrinsics_paths:
        with open(path, "r") as f:
            intrinsics = json.load(f)
        intrinsics_list.append(
            [
                intrinsics["width"],
                intrinsics["height"],
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["ppx"],
                intrinsics["ppy"],
            ]
        )
    return intrinsics_list


# Timestamp-based frame alignment
def align_timestamps(config, rti_config=None, **kwargs):
    datapaths = [
        os.path.join(config.datafolder, f"camera_{view}", config.input_filetype)
        for view in range(config.num_cameras)
    ]
    config.datapaths = datapaths
    if config.saved_timestamps_path:
        return np.load(config.saved_timestamps_path)

    filetype1 = kwargs.get("filetype1", "npy")
    filetype2 = kwargs.get("filetype2", "npy")
    eps = kwargs.get("eps", 1e6)

    if rti_config is None:
        timestamps = [
            linear_timescale(
                path1=config.datapaths[0],
                ref1=config.start_end_ref[0],
                filetype1=filetype1,
            )
        ]
        for idx in range(1, len(datapaths)):
            timestamps.append(
                linear_timescale(
                    path1=config.datapaths[0],
                    ref1=config.start_end_ref[0],
                    path2=config.datapaths[idx],
                    ref2=config.start_end_ref[idx],
                    filetype1=filetype1,
                    filetype2=filetype2,
                    eps=eps,
                )
            )
    else:
        timestamps = []
        for idx in range(len(datapaths)):
            timestamps.append(
                linear_timescale(
                    path1=rti_config.rti_datapath,
                    ref1=rti_config.rti_start_end_ref[0],
                    path2=datapaths[idx],
                    ref2=config.start_end_ref[idx],
                    filetype1=filetype1,
                    filetype2=filetype2,
                    eps=eps,
                )
            )

    timestamps = np.array(timestamps)

    if np.isnan(timestamps).any():
        print("NaN present")
        keep = ~np.any(np.isnan(timestamps), axis=0)
        timestamps = timestamps[:, keep]
        kept_indices = np.where(keep)[0][0]

    if config.path_to_save_timestamps:
        np.save(config.path_to_save_timestamps, timestamps.T)
        np.save(config.kept_indices_path, kept_indices)

    return timestamps.T


def logger(log_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                with open(log_file, "a") as f:
                    filename = kwargs.get("filename", "Unknown File")
                    f.write(f"Failed Case: {filename}\n")
                    f.write("Error Traceback:\n")
                    f.write(traceback.format_exc())
                    f.write("\n")

                return None, None

        return wrapper

    return decorator
