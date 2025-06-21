# Camera-based Ground Truth Collection (for Radio Tomographic Imaging).

### Environment 
- Install all dependencies in `requirements.txt` in a virtual environment. 

### Data Collection
- run ` realsense.py ` to collect synchronized Point Clouds and RGB frames.

### For Person-Tracking
- run ` utils.annotate_points.py ` to select the Corner Keypoints from any reference Frame. 
- run ` utils.compute_homography.py ` to get the Transformation Matrix for Bird Eye View. *Need the 4 Corner Coordinates to define the Plane.   
- run ` pose.py ` to collect midpoint of feet as position data, based on YOLOv8 model. 

### For GT
- run `utils.npy_to_ply` to convert raw `.npy` files into `.ply` objects. 
- run `utils.vis_pcd.py` to select the Corner Keypoints from any reference Frame.
- Ensure that the order of camera positions in `main.py` match `realsense_data/camera_{view}`. 
- run ` main.py ` to generate Ground Truth `(images/gt)` and Uncertainty Masks `(images/cmask)` after data collection. 

### Misc

Key Ideas:
- Super-resolution - To distinguish seperate objects close to each other
- Object Shape Definition - To learn exact shape of objects
- Node Reduction - Fewer Nodes
- 3D Reconstruction - Multi-splice model
- Time-series - For object recognition and tracking

Camera Setup: Intel Realsense D435F or D455

> References for RealSense Python Wrapper: 
> - https://pypi.org/project/pyrealsense2/
> - https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
