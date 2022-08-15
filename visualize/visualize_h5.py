import os
import numpy as np
import open3d as o3d
import h5py

from scipy.io import loadmat


VISUALIZE = True
COMPUTE_NORMALS = True

free_dir = "/home/nicolai/phd/code/3DShapeGen/gen_sdf/02691156/"

names = sorted(os.listdir(free_dir))
for name in names:
    point_file = h5py.File(os.path.join(free_dir, name, "ori_sample.h5"))
    point_cloud = point_file["pc_sdf_sample"][:]
    points, sdf = point_cloud[:, :3], point_cloud[:, -1]

    if COMPUTE_NORMALS:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[np.where(sdf == 0.0)])
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    if VISUALIZE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.zeros_like(points)
        colors[np.where(sdf > 0)] = [1, 0, 0]
        colors[np.where(sdf <= 0)] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
