import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import Config, GT_Config, RTI_Config
from masks.format_pcd import *
from masks.frame_cmask import make_buffer_mask, make_final_cmask
from masks.helpers import align_timestamps, prepare_intrinsics
from utils.npy_to_ply import npy_to_ply


def test_cases(timestamps, filenames):
    indices = []
    for name in filenames:
        indices.append(np.where(timestamps[:, 0] == name)[0][0])
    return timestamps[indices]


def precompute_constants(config):
    affine_matrices = []
    DOI_planes = []
    for view in range(config.num_cameras):
        picked_points = np.load(config.picked_points_paths[view])
        corners, vh, polygon, centroid = define_plane(picked_points)
        DOI_planes.append([corners, vh, polygon, centroid])
        affine_matrices.append(affine_matrix(corners, config.dst_pts))
        # plot_corners(transform_pts(corners, affine_matrices[view]))
    return DOI_planes, affine_matrices


def create_dataset(config, timestamps):
    os.makedirs(os.path.join(config.gt_dir, config.datafolder), exist_ok=True)
    os.makedirs(os.path.join(config.cmask_dir, config.datafolder), exist_ok=True)
    os.makedirs(os.path.join(config.plt_dir, config.datafolder), exist_ok=True)

    DOI_planes, affine_matrices = precompute_constants(config)
    buffer_mask = make_buffer_mask(config.image_size)

    block = 5000 if not config.save_masks else 1
    rows = tqdm(timestamps)
    for row in rows:
        all_points = []
        filename = str(row[0])
        for view in range(config.num_cameras):
            intrinsics = config.intrinsics_list[view]
            # fix depth to npy
            npypath = os.path.join(
                config.datapaths[view], str(row[view]) + "." + config.input_filetype
            )

            pcd = npy_to_ply(npypath, intrinsics, save=config.save_raw_pcd)
            cropped_points = crop_PCD(pcd, *DOI_planes[view])  # 3D to 2D points
            points = flatten(
                affine_matrices[view],
                cropped_points,
                DOI_size=config.DOI_size,
                buffer=GT_Config.buffer,
                image_size=config.image_size,
            )
            all_points.append(points)

        gt, cmask = make_final_cmask(
            all_points,
            cameras=config.cameras,
            buffer_mask=buffer_mask,
            filename=filename,
            object_count=config.object_count,
            image_size=config.image_size,
            plot=config.plot,
        )

        if not isinstance(gt, np.ndarray):
            print(f"Error in File: {filename}")
            continue

        if config.save_masks:
            gt_path = os.path.join(
                config.gt_dir,
                config.datafolder,
                filename + "." + config.output_filetype,
            )
            cmask_path = os.path.join(
                config.cmask_dir,
                config.datafolder,
                filename + "." + config.output_filetype,
            )
            # cv2 and np set (0, 0) at top left by default (origin upper).
            cv2.imwrite(gt_path, np.flipud(gt) * 255)
            cv2.imwrite(cmask_path, np.flipud(cmask) * 255)
        rows.set_description(f"Prepared GT for {config.datafolder}/{row[0]}")

        if config.save_2D_points:
            plt_path = os.path.join(
                config.plt_dir,
                config.datafolder,
                filename + "." + config.output_filetype,
            )

            view_colors = ["red", "green", "blue", "orange"]
            view_labels = ["View 0", "View 1", "View 2", "View 3"]
            for i, points in enumerate(all_points):
                x, y = points[:, 0], points[:, 1]
                plt.scatter(
                    x, y, s=1, color=view_colors[i], label=view_labels[i], alpha=0.7
                )

            plt.title("2D Projected Points from Each View")
            plt.xlim((0, config.image_size))
            plt.ylim((0, config.image_size))
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(plt_path)
            plt.close()

        cv2.imshow("", gt * 255)
        key = cv2.waitKey(block)
        if key == 27:  #  ESC to quit
            break

    cv2.destroyAllWindows()


def main():
    config = Config(
        datafolder="realsense_data_306_b",
        gt_dir=os.path.join("images", "gt"),
        cmask_dir=os.path.join("images", "cmask"),
        plt_dir=os.path.join("images", "plt"),
        object_count=2,  # objects in DOI
        image_size=112,  # in pixels
        DOI_size=3,  # DOI square length in meters
    )

    # Camera positions [x, y] in global coordinates
    # Must ensure camera_{k} is k-th entry in cameras
    # cameras = np.array([[3.73, 1.5], [1.5, 3.8], [-0.6, 1.5], [1.5, -0.85]])
    cameras = np.array([[3.73, 1.5], [1.5, -0.85], [-0.6, 1.5], [1.5, 3.8]])
    cameras = cameras * (config.image_size / config.DOI_size)
    config.cameras = np.round(cameras, 1)

    # Corners of DOI plane in meters, ensure correct order
    config.dst_pts = np.array([[3, 3], [0, 3], [0, 0], [3, 0]])

    rti_config = RTI_Config()
    # Reference start and end frames
    config.start_end_ref = {
        0: [56863160105, 59380817823],
        1: [56863170676, 59380847000],
        2: [56836339960, 59354071415],
        3: [56836362644, 59354092562],
    }
    # config.path_to_save_timestamps = "timestamps.npy"
    # config.kept_indices_path = "kept_indices.npy"
    timestamps = align_timestamps(config, rti_config)
    # Picked Points Path (Corners)
    config.picked_points_paths = [
        os.path.join("constants", f"picked_points_{view}.npy")
        for view in range(config.num_cameras)
    ]

    # Intrinsics Path
    intrinsics_paths = [
        os.path.join(
            config.datafolder, f"camera_{view}", "intrinsics", "intrinsics.json"
        )
        for view in range(config.num_cameras)
    ]
    config.intrinsics_list = prepare_intrinsics(intrinsics_paths)

    # Comment following block for debugging.
    # timestamps = align_timestamps(config)
    # print(timestamps.shape)
    # config.datafolder = "realsense_data_306_b"
    # config.gt_dir = os.path.join("test", "gt")
    # config.cmask_dir = os.path.join("test", "cmask")
    # config.plt_dir = os.path.join("test", "plt")
    # config.plot = True
    # filenames = [
    #     56880026854,
    #     56995060462,
    #     56992593854,
    #     57375961923,
    #     57032290327,
    # ]
    # timestamps = test_cases(timestamps, filenames)

    config.save_masks = True
    config.save_2D_points = True
    create_dataset(config, timestamps)

    return


if __name__ == "__main__":
    main()
