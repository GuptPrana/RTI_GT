import cv2
import numpy as np


def compute_homography(
    src_pts, show_img=True, img_src_pth="", save_H=True, save_H_pth=""
):
    x = 256
    y = 256
    b = 16

    delta_x = (x - 2 * b) / 2
    delta_y = (y - 2 * b) / 2
    _dst_pts = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])

    dst_pts = np.zeros_like(_dst_pts, dtype=np.float32)
    dst_pts[:, 0] = _dst_pts[:, 0] * delta_x + b
    dst_pts[:, 1] = _dst_pts[:, 1] * delta_y + b
    # dst_pts = delta_x * dst_pts + b

    H, _ = cv2.findHomography(src_pts, dst_pts)

    if save_H:
        np.save(save_H_pth, H)

    if show_img:
        img_src = cv2.imread(img_src_pth)
        img_out = cv2.warpPerspective(img_src, H, (x, y))
        # (img_src.shape[1],img_src.shape[0]))
        # cv2.imwrite("../realsense_data/warped_image.jpg", img_out)
        cv2.imshow("Source Image", img_src)
        cv2.imshow("Warped Image", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return H


if __name__ == "__main__":
    cam_view = 1
    img_name = "temp.jpg"

    src_pts_pth = "../constants/source_points.npy"
    img_src_pth = f"../realsense_data/cam{cam_view}/{img_name}"
    save_H_pth = f"../constants/transform_{cam_view}.npy"
    src_pts = np.load(src_pts_pth)

    H = compute_homography(src_pts, img_src_pth=img_src_pth, save_H_pth=save_H_pth)
    print(H)
