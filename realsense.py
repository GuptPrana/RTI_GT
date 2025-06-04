import datetime
import json
import os

import cv2
import numpy as np
import pyrealsense2 as rs


def capture(NUM_CAMERAS, SAVE_DIR):
    pipelines = []
    configs = []

    # Prepare directories
    for cam_idx in range(NUM_CAMERAS):
        cam_dir = os.path.join(SAVE_DIR, f"camera_{cam_idx+1}")
        os.makedirs(cam_dir, exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "intrinsics"), exist_ok=True)

    # Detect connected RealSense devices
    context = rs.context()
    connected_devices = [
        context.devices[i].get_info(rs.camera_info.serial_number)
        for i in range(len(context.devices))
    ]

    if len(connected_devices) < NUM_CAMERAS:
        raise Exception(
            f"Only {len(connected_devices)} out of {NUM_CAMERAS} cameras connected."
        )

    for i in range(NUM_CAMERAS):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(connected_devices[i])
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipelines.append(pipeline)
        configs.append(config)
        pipelines[i].start(configs[i])

    print("Press 'q' to stop streaming.")

    # Save camera intrinsics
    def save_camera_intrinsics(intrinsics, file_path):
        intrinsics_data = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "model": str(intrinsics.model),
            "coeffs": intrinsics.coeffs,
        }
        with open(file_path, "w") as f:
            json.dump(intrinsics_data, f, indent=4)

    for i in range(NUM_CAMERAS):
        intrinsics = (
            pipelines[i]
            .get_active_profile()
            .get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        intrinsics_file = os.path.join(
            SAVE_DIR, f"camera_{i+1}", "intrinsics", "intrinsics.json"
        )
        save_camera_intrinsics(intrinsics, intrinsics_file)

    frame_count = 0
    try:
        while True:
            frames = []

            for i in range(NUM_CAMERAS):
                frame_set = pipelines[i].wait_for_frames()
                frames.append(frame_set)

            for i, frame_set in enumerate(frames):
                depth_frame = frame_set.get_depth_frame()
                color_frame = frame_set.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Use local system time as timestamp: DDHHMMSSmmmmmm
                now = datetime.datetime.now()
                timestamp_str = now.strftime("%d%H%M%S%f")

                rgb_filename = os.path.join(
                    SAVE_DIR, f"camera_{i+1}", "rgb", f"{timestamp_str}.jpg"
                )
                depth_filename = os.path.join(
                    SAVE_DIR, f"camera_{i+1}", "depth", f"{timestamp_str}.npy"
                )
                cv2.imwrite(rgb_filename, color_image)
                np.save(depth_filename, depth_image)

                print(f"Saved Camera {i+1} - Frame {frame_count}: {timestamp_str}")

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for i in range(NUM_CAMERAS):
            pipelines[i].stop()

    return


if __name__ == "main":
    NUM_CAMERAS = 2
    SAVE_DIR = "realsense_data"
    capture(NUM_CAMERAS, SAVE_DIR)
