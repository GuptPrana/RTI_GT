# Camera-based Ground Truth Collection (RTI).

### Data Collection
- run ` realsense.py ` to collect synchronized Point Clouds and RGB frames.

### For Person-Tracking
- run ` utils/annotate_points.py ` to select the Corner Keypoints from any reference Frame. 
- run ` utils/compute_homography.py ` to get the Transformation Matrix for Bird Eye View.  
  **Need the 4 Corner Coordinates to define the Plane.   
- run ` pose.py ` to collect midpoint of feet as position data, based on YOLOv8 model. 

### For GT
- run `utils/npy_to_ply.py` to convert raw `.npy` file(s) into `.ply` object(s) for selecting Keypoints.
- run `utils/vis_pcd.py.py` to select the Corner Keypoints from any reference Frame.
- Ensure that the order of camera positions in `main.py` match `realsense_data/camera_{view}`. 
- run ` main.py ` to generate Ground Truth `(images/gt)` and Uncertainty Masks `(images/cmask)` after data collection. 

### Environment
- Create a virtual environment and activate it before running any scripts.
- Install all dependencies in `requirements_PT.txt` for Person-Tracking. 
- Install all dependencies in `requirements_GT.txt` for Depth-based Ground Truth.
- Below is an example using Python's built-in venv on CLI:
  ```bash
  # Create the virtual environment:
  python -m venv environment_name
  # Activate in Windows:
  environment_name/Scripts/activate
  # Activate in Mac/Linux:
  source environment_name/bin/activate
  # Install dependencies (e.g. for GT)
  pip install -r requirements_GT.txt
  # To deactivate:
  deactivate
  # To delete environment:
  rm -r environment_name
  ```
  
### Misc
Camera Setup: Intel Realsense D435(F) or D455(F)

> References for RealSense Python Wrapper: 
> - https://pypi.org/project/pyrealsense2/
> - https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python

> Useful Tools for Visualization and Calibration
> - https://www.intelrealsense.com/sdk-2/#sdk2-tools
> - https://dev.intelrealsense.com/docs/self-calibration-for-depth-cameras