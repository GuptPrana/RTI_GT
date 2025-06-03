# Camera-based Ground Truth Collection (for Radio Tomographic Imaging).

### How to Use

- run ` python realsense.py ` to collect synchronized Point Clouds and RGB frames.
- run ` python getPointsByAnnotation.py ` to select the Corner Keypoints from any reference Frame. 
- run ` python computeHomography.py ` to get the Transformation Matrix for Bird Eye View. *Need the 4 Corner Coordinates to define the Plane.   
- run ` python pose.py ` to collect midpoint of feet as position data, based on YOLOv8 model. 
- run ` python convertKeypoints.py ` to convert raw Coordinates from Camera View to Plane Coordinates by applying the Transformation Matrix. 
- run ` python getPCDSplice.py ` to Align Point Clouds and Extract Plane Splice. 
- run ` python finalGT.py ` to Merge Splices and Generate Ground Truth Image and Binary Mask.

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
