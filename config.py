import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Config:
    datafolder: str
    object_count: int
    gt_dir: str
    cmask_dir: str
    plt_dir: str
    input_filetype: str = "npy"
    output_filetype: str = "jpg"
    save_masks: bool = False  # save GT and cmask
    save_raw_pcd: bool = False  # raw .ply file
    save_DOI_pcd: bool = False  # cropped DOI points
    save_2D_points: bool = False  # post flatten()
    plot: bool = False  # see all masks (for debugging)

    num_cameras: int = 4
    image_size: int = 224
    DOI_size: int = 3
    cameras: Optional[np.ndarray] = None
    intrinsics_list: Optional[list] = None
    dst_pts: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([[3, 3], [0, 3], [0, 0], [3, 0]])
    )  # DOI Corners

    saved_timestamps_path: Optional[str] = None  # precomputed cam timestamps
    start_end_ref: Optional[Dict[int, List[int]]] = field(
        default_factory=lambda: {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1]}
    )
    input_datapath: Optional[str] = None  # RTI timestamps
    path_to_save_timestamps: Optional[str] = None  # path to save cam timestamps
    kept_indices_path: Optional[str] = (
        None  # path to save kept indices from align_timestamp
    )
    picked_points_paths: Optional[list] = None
    datapaths: Optional[list] = None


@dataclass
class RTI_Config:
    rti_datapath: str = os.path.join("makexrti", "xpra_im", "testh")
    rti_start_end_ref: Optional[Dict[int, List[int]]] = field(
        default_factory=lambda: {0: [56836008251, 59352486034]}
    )


@dataclass
class GT_Config:
    alpha: Optional[float] = None  # alphashape
    minpoints: int = 10  # minpoints for object_shape
    spatial_filter_rad: float = 3.0  # nn filter radius
    spatial_filter_neigh: int = 25  # nn filter neighbors
    GMM_thresh: float = 0.95
    eps: float = 3.0  # shadow ray angle jitter
    buffer: int = 10  # cmask side buffer (22)
    corner_mult: int = 3  # cmask corner buffer (4)
