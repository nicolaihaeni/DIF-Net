import os
import numpy as np
import open3d as o3d
import h5py

import torch
from pytorch3d.ops import sample_farthest_points


VISUALIZE = False
COMPUTE_NORMALS = False

free_dir = "/home/nicolai/phd/code/3DShapeGen/gen_sdf/02691156/"

names = sorted(os.listdir(free_dir))
for name in names:
    point_file = h5py.File(os.path.join(free_dir, name, "ori_sample.h5"))

    free_pts = point_file["free_pts"][:]
    surface_pts = point_file["surface_pts"][:]
    points, normals, sdf = np.split(surface_pts, [3, 6], axis=-1)
    free_points, free_sdf = np.split(free_pts, [3], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    colors = np.zeros_like(points)
    colors[:, 0] = 1
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if COMPUTE_NORMALS:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    free_pcd = o3d.geometry.PointCloud()
    free_pcd.points = o3d.utility.Vector3dVector(free_points)
    colors = np.zeros_like(free_points)
    colors[:, 1] = 1
    free_pcd.colors = o3d.utility.Vector3dVector(colors)
    if VISUALIZE:
        o3d.visualization.draw_geometries([pcd, free_pcd])

    # Downsample and do farthest point sampling
    down_sampled_pcd = o3d.geometry.PointCloud()
    down_sampled_pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros_like(points)
    colors[:, 2] = 1
    down_sampled_pcd.colors = o3d.utility.Vector3dVector(colors)
    down_sampled_pcd = down_sampled_pcd.voxel_down_sample(voxel_size=0.01)
    down_sampled_pcd = down_sampled_pcd.translate((1.0, 0.0, 0.0))

    # Do farthest point sampling
    down_sampled_points = np.asarray(down_sampled_pcd.points)
    farthest_points, idx = sample_farthest_points(
        torch.tensor(down_sampled_points[None]).cuda(), K=1024
    )
    farthest_points = farthest_points.squeeze().cpu().detach().numpy()
    farthest_pcd = o3d.geometry.PointCloud()
    farthest_pcd.points = o3d.utility.Vector3dVector(farthest_points)
    colors = np.zeros_like(farthest_points)
    colors[:, 1] = 1
    farthest_pcd.colors = o3d.utility.Vector3dVector(colors)
    farthest_pcd = farthest_pcd.translate((1.0, 0.0, 0.0))

    print(down_sampled_points.shape[0])
    if down_sampled_points.shape[0] < 1024:
        print(f"Too few points in: {name}")
    # o3d.visualization.draw_geometries([pcd, down_sampled_pcd, farthest_pcd])
