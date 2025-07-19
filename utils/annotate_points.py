import os
import cv2
import numpy as np

clicked_points = []


def click_event(event, x, y, img, flags):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(x, " ", y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + "," + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow("Image", img)


if __name__ == "__main__":
    cam_view = 1
    data_folder = "realsense_data"
    img_name = "temp.jpg"  # os.listdir(f"{data_folder}/cam{cam_view}/rgb")[0]
    src_pts_path = f"constants/source_points_{cam_view}.npy"

    img = cv2.imread(f"{data_folder}/cam{cam_view}/rgb/{img_name}")
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.save(src_pts_path, np.array(clicked_points))
