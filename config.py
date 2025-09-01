import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Config:
    datafolder: str
    object_count: int
    save_raw_pcd: bool = False  # raw .ply file
    save_DOI_pcd: bool = False  # cropped DOI points
    see_2D_points: bool = False  # post flatten()
    plot: bool = False
    num_cameras: int = 4
    image_size: int = 224
    DOI_size: int = 3
    buffer: int = 22
    object_alpha: Optional[Dict[str, float]] = None
    gt_dir: str = "images/gt"
    cmask_dir: str = "images/cmask"
    plt_dir: str = "images/plt"
    input_filetype: str = "npy"
    output_filetype: str = "jpg"
    intrinsics_list: Optional[list] = None
    cameras: Optional[np.ndarray] = None
    saved_timestamps_path: Optional[str] = None  # precomputed cam timestamps
    cam_split_by_pc: Optional[Dict[int, List[int]]] = field(
        default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0}
    )  # cam_idx: pc_idx
    start_end_ref: Optional[Dict[int, List[int]]] = field(
        default_factory=lambda: {0: [0, -1], 1: [0, -1], 2: [0, -1], 3: [0, -1]}
    )
    input_datapath: Optional[str] = None  # RTI timestamps
    path_to_save_timestamps: Optional[str] = None  # path to save cam timestamps
    dst_pts: Optional[np.ndarray] = field(
        default_factory=lambda: np.array([[3, 3], [0, 3], [0, 0], [3, 0]])
    )  # DOI Corners
    picked_points_paths: Optional[list] = None
    datapaths: Optional[list] = None


@dataclass
class RTI_Config:
    rti_datapath: str = os.path.join("makexrti", "xpra_im", "testh")
    rti_start_end_ref: Optional[Dict[int, List[int]]] = field(
        default_factory=lambda: {0: [56836008251, 59352977821]}
    )
