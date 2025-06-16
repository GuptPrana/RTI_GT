import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def colormap(plypath):
    depth = np.load(plypath) / 1000.0  # from mm to meters

    depth_trunc = 8.0
    depth[depth > depth_trunc] = np.nan

    # Plot truncated depth
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap="jet", vmin=0, vmax=depth_trunc)
    plt.colorbar(label="Depth (m)")
    plt.title(f"Depth Visualization")
    plt.axis("off")
    plt.show()


def seePCD(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    o3d.visualization.draw_geometries([pcd], window_name="PCD")


def selectpointsPCD(plypath):
    pcd = o3d.io.read_point_cloud(plypath)
    print("Select Points with Shift + Left Click, Press 'q' to Exit.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_points_indices = vis.get_picked_points()
    picked_points = np.asarray(pcd.points)[picked_points_indices]
    print("Picked points:\n", picked_points)

    return picked_points
