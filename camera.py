import numpy as np
# from numpy.linalg import inv

class Camera:
  def __init__(self, roll, pitch, yaw, X, Y, Z):
    self.phi = roll
    self.theta = pitch
    self.psi = yaw
    self.translation = [X, Y, Z]

  def rotation_matrix(self):
    R = np.array([[1, 0, 0],
                  [0, np.cos(self.phi), -np.sin(self.phi)],
                  [0, np.sin(self.phi), np.cos(self.phi)]])
    P = np.array([[np.cos(self.theta), 0, np.sin(self.theta)],
                  [0, 1, 0],
                  [-np.sin(self.theta), 0, np.cos(self.theta)]])
    Y = np.array([[np.cos(self.psi), -np.sin(self.psi), 0],
                  [np.sin(self.psi), np.cos(self.psi), 0],
                  [0, 0, 1]])    
    return R @ P @ Y
  
  def transformation_matrix(self):
    transform = np.column_stack((self.rotation_matrix(), np.array(self.translation)))
    return np.vstack((transform, np.array([0, 0, 0, 1])))

  def inverse_transformation_matrix(self):
    # return inv(self.transformation_matrix())
    rotation_t = self.rotation_matrix().transpose()
    translation_t = - self.rotation_matrix().transpose() @ np.array(self.translation)
    return np.vstack((np.column_stack((rotation_t, translation_t)), np.array([0, 0, 0, 1]))) 

  def transform_pointcloud(self, pointcloud, transformation_matrix):
    # transformation_matrix = C1.transformation_matrix() to transform pointcloud to C1 coordinate space. 
    points = np.asarray(pointcloud.points)
    # Homogeneous Coordinates: [X, Y, Z, 1]
    homogeneous_points = np.column_stack((points, np.ones((points.shape[0], 1))))
    inverse_transformed = (self.inverse_transformation_matrix() @ homogeneous_points.transpose()).transpose()
    transformed_points = (transformation_matrix @ inverse_transformed.transpose()).transpose()[:, :3]
    # print(transformation_matrix)
    # print(self.inverse_transformation_matrix())
    # Need to read the array in o3d.geometry.PointCloud() object
    return transformed_points

  ## Can add intrinsic calibration codes here too