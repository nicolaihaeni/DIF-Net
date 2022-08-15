import os
import numpy as np
import open3d as o3d

from scipy.io import loadmat


VISUALIZE = True

free_dir = "/home/nicolai/phd/code/DIF-Net/datasets/plane/free_space_pts"
sdf_dir = "/home/nicolai/phd/code/DIF-Net/datasets/plane/surface_pts_n_normal"
output_dir = "/home/nicolai/phd/code/DIF-Net/datasets/plane/"

names = sorted(os.listdir(free_dir))
for name in names:
    free_points = loadmat(os.path.join(free_dir, name))
    free_points = free_points["p_sdf"]

    point_cloud = loadmat(os.path.join(sdf_dir, name))
    point_cloud = point_cloud["p"]

    points = np.concatenate([point_cloud[:, :3], free_points[:, :3]])
    sdf = np.concatenate([point_cloud[:, -1], free_points[:, -1]])
    normals = np.concatenate(
        [point_cloud[:, 3:], np.ones_like(free_points[:, :3]) * -1]
    )

    if VISUALIZE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[np.where(sdf <= 0.0)])
        colors = np.zeros_like(points)
        colors[np.where(sdf > 0)] = [1, 0, 0]
        colors[np.where(sdf <= 0)] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors[np.where(sdf <= 0.0)])
        o3d.visualization.draw_geometries([pcd])
