import os

import cv2
import numpy as numpy


def generate_video(image_dir, output):
    imgs = os.listdir(image_dir)
    fps = 30

    frame = cv2.imread(os.path.join(image_dir, imgs[0]))
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output, fourcc, fps, (w, h))

    for i in imgs:
        frame = cv2.imread(os.path.join(image_dir, i))
        video.write(frame)

    video.release()
    print(f"Saved {output}")


if __name__ == "__main__":
    image_dir = os.path.join("images", "gt", "realsense_data_306_b")
    output = os.path.join("assets", "gt_example.mp4")
    generate_video(image_dir, output)
