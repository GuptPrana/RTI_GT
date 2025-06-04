import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import os
import json

NUM_CAMERAS = 2

pipelines = []
configs = []

SAVE_DIR = "realsense_data"

for cam_idx in range(NUM_CAMERAS):
    cam_dir = os.path.join(SAVE_DIR, f"camera_{cam_idx+1}")
    os.makedirs(cam_dir, exist_ok=True)
    os.makedirs(os.path.join(cam_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(cam_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(cam_dir, "intrinsics"), exist_ok=True)

context = rs.context()
connected_devices = [context.devices[i].get_info(rs.camera_info.serial_number) for i in range(len(context.devices))]

if len(connected_devices) < NUM_CAMERAS:
    raise Exception(f"Only {len(connected_devices)} RealSense cameras found. Expected {NUM_CAMERAS}.")

print("Connected RealSense Cameras:", connected_devices[:NUM_CAMERAS])

for i in range(NUM_CAMERAS):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(connected_devices[i])

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)  
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15) 

    pipelines.append(pipeline)
    configs.append(config)

for i in range(NUM_CAMERAS):
    pipelines[i].start(configs[i])

print("Streaming! Press 'q' to stop.")

def save_camera_intrinsics(intrinsics, file_path):
    intrinsics_data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model),  # Convert distortion model to string
        "coeffs": intrinsics.coeffs,
    }
    with open(file_path, 'w') as f:
        json.dump(intrinsics_data, f, indent=4)


# Save intrinsics for each camera
for i in range(NUM_CAMERAS):
    intrinsics = pipelines[i].get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    intrinsics_file = os.path.join(SAVE_DIR, f"camera_{i+1}", "intrinsics", "intrinsics.json")
    save_camera_intrinsics(intrinsics, intrinsics_file)
    print(f"Saved intrinsics for Camera {i+1}: {intrinsics_file}")

frame_count = 0
try:
    while True:
        frames = []
        timestamps = []

        # Fetch frames from each camera
        for i in range(NUM_CAMERAS):
            frame_set = pipelines[i].wait_for_frames()
            timestamps.append(frame_set.get_timestamp())  # Capture timestamp for synchronization
            frames.append(frame_set)


        for i, frame_set in enumerate(frames):
            depth_frame = frame_set.get_depth_frame()
            color_frame = frame_set.get_color_frame()

            if not depth_frame or not color_frame:
                continue  

           
            depth_image = np.asanyarray(depth_frame.get_data())  # 16-bit depth
            color_image = np.asanyarray(color_frame.get_data())  # 8-bit RGB

           
            timestamp = int(timestamps[i]) 
            rgb_filename = os.path.join(SAVE_DIR, f"camera_{i+1}", "rgb", f"{timestamp}.jpg")
            depth_filename = os.path.join(SAVE_DIR, f"camera_{i+1}", "depth", f"{timestamp}.npy")
            cv2.imwrite(rgb_filename, color_image) 
            np.save(depth_filename, depth_image)  

            print(f"Saved Camera {i+1} - Frame {frame_count}: {rgb_filename}, {depth_filename}")

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    for i in range(NUM_CAMERAS):
        pipelines[i].stop()

    print("Streaming stopped. All frames saved.")