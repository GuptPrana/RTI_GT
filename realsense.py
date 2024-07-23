import pyrealsense2 as rs
import datetime
import numpy as np
import cv2
import os

pipeline = rs.pipeline()
config = rs.config()

# 6FPS seems to be lowest 
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
# for depth, rs.stream.depth
pipeline.start(config)
cwd = os.getcwd()

while True: 
    frame_object = pipeline.wait_for_frames()
    # for depth image, frame_object.get_depth_frame()
    color_frame = frame_object.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    
    # save frame with local timestamp
    timestamp = datetime.datetime.now().time()
    # hh:mm:ss.xxxxxx
    filename = str(timestamp).replace(':', '_').replace('.', '_') + '.jpg'
    filepath = os.path.join(cwd, 'data', filename)

    # need BGR for cv2
    cv2.imwrite(filepath, color_image)
    cv2.imshow('RGB', color_image)

    # press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()