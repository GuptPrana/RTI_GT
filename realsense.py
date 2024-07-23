import pyrealsense2 as rs
import datetime
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

# for depth, rs.stream.depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

while True: 
    frame_object = pipeline.wait_for_frame()
    # for depth image, frame_object.get_depth_frame()
    color_frame = frame_object.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    
    # save frame with local timestamp
    timestamp = datetime.datetime.now().time()
    # hh:mm:ss.xxxxxx
    filename = 'data/' + str(timestamp).replace('.', ':') + '.jpg'

    # need BGR for cv2
    cv2.imwrite(filename, color_image)
    cv2.imshow('RGB', color_image)

    # press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()