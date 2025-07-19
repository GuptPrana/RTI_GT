import datetime
import json
import os

import cv2
import numpy as np
import pyrealsense2 as rs


def capture(num_cameras, save_dir):
    pipelines = []
    configs = []

    # Prepare directories
    for cam_idx in range(num_cameras):
        cam_dir = os.path.join(save_dir, f"camera_{cam_idx}")
        os.makedirs(cam_dir, exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "npy"), exist_ok=True)
        os.makedirs(os.path.join(cam_dir, "intrinsics"), exist_ok=True)

    # Detect connected RealSense devices
    context = rs.context()
    connected_devices = [
        context.devices[i].get_info(rs.camera_info.serial_number)
        for i in range(len(context.devices))
    ]

    if len(connected_devices) < num_cameras:
    if len(connected_devices) < num_cameras:
        raise Exception(
            f"Only {len(connected_devices)} out of {num_cameras} cameras connected."
            f"Only {len(connected_devices)} out of {num_cameras} cameras connected."
        )

    # Can fill camera serial number to order cam_views
    ordered_serials = [
        "",  # camera_0
        "",  # camera_1
        "",  # camera_2
        "",  # camera_3
    ]

    for i in range(num_cameras):
    for i in range(num_cameras):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(
            ordered_serials[i]
        )  # config.enable_device(connected_devices[i]) if no ordered_serials
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipelines.append(pipeline)
        configs.append(config)
        pipelines[i].start(configs[i])

    print("Press 'q' to stop streaming.")

    # spatial filtering
    # decimation = rs.decimation_filter()
    # decimation.set_option(rs.option.filter_magnitude, 2)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)  # 1–5
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # 0–1
    # spatial.set_option(rs.option.filter_smooth_delta, 20)    # 1–100
    spatial.set_option(rs.option.holes_fill, 3)  # 0–5

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

    for i in range(num_cameras):
    for i in range(num_cameras):
        intrinsics = (
            pipelines[i]
            .get_active_profile()
            .get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        intrinsics_file = os.path.join(
            save_dir, f"camera_{i}", "intrinsics", "intrinsics.json"
        )
        save_camera_intrinsics(intrinsics, intrinsics_file)

    frame_count = 0
    try:
        while True:
            frames = []

            for i in range(num_cameras):
                frame_set = pipelines[i].wait_for_frames()
                frames.append(frame_set)

            for i, frame_set in enumerate(frames):
                depth_frame = frame_set.get_depth_frame()
                color_frame = frame_set.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # spatial filtering
                # depth_frame = decimation.process(depth_frame)
                depth_frame = spatial.process(depth_frame)

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Use local system time as timestamp: DDHHMMSSmmmmmm
                now = datetime.datetime.now()
                timestamp_str = now.strftime("%d%H%M%S%f")

                rgb_filename = os.path.join(
                    save_dir, f"camera_{i}", "rgb", f"{timestamp_str}.jpg"
                )
                depth_filename = os.path.join(
                    save_dir, f"camera_{i}", "npy", f"{timestamp_str}.npy"
                )
                cv2.imwrite(rgb_filename, color_image)
                np.save(depth_filename, depth_image)

                print(f"Saved Camera {i} - Frame {frame_count}: {timestamp_str}")

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for i in range(num_cameras):
        for i in range(num_cameras):
            pipelines[i].stop()

    return


if __name__ == "__main__":
    num_cameras = 2
    save_dir = "realsense_data"
    os.makedirs(save_dir, exist_ok=True)
    # add logic to save ply directly
    capture(num_cameras, save_dir)
