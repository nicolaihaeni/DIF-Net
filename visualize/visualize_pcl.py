import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
import torch
from scipy.io import loadmat
from pytorch3d.ops import sample_farthest_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surf_file_path",
        required=True,
        type=str,
        help="abs path to surface point cloud",
    )
    parser.add_argument(
        "--free_file_path",
        required=True,
        type=str,
        help="abs path to point cloud",
    )
    parser.add_argument(
        "--vis_normals",
        default=False,
        action="store_true",
        help="visualize point normals",
    )

    args = parser.parse_args()
    print(f"Visualizing {args.surf_file_path}")

    point_set = loadmat(args.surf_file_path)
    surf_points = point_set["p"][:, :3]
    surf_normals = point_set["p"][:, 3:]
    surf_colors = np.zeros_like(surf_points)
    surf_colors[:, 0] = 1

    farthest_points = sample_farthest_points(
        torch.tensor(surf_points).unsqueeze(0).cuda(), K=2048, random_start_point=True
    )
    farthest_points = farthest_points[0].squeeze().cpu().numpy()
    farthest_colors = np.zeros_like(farthest_points)
    farthest_colors[:, -1] = 1
    farthest_pcd = o3d.geometry.PointCloud()
    farthest_pcd.points = o3d.utility.Vector3dVector(farthest_points)
    farthest_pcd.colors = o3d.utility.Vector3dVector(surf_colors)

    # point_set = loadmat(args.free_file_path)
    # free_points = point_set["p_sdf"][:, :3]
    # free_colors = np.zeros_like(free_points)
    # free_colors[:, 1] = 1

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    surf_pcd = o3d.geometry.PointCloud()
    surf_pcd.points = o3d.utility.Vector3dVector(surf_points)
    surf_pcd.normals = o3d.utility.Vector3dVector(surf_normals)
    surf_pcd.colors = o3d.utility.Vector3dVector(surf_colors)
    # free_pcd = o3d.geometry.PointCloud()
    # free_pcd.points = o3d.utility.Vector3dVector(free_points)
    # free_pcd.colors = o3d.utility.Vector3dVector(free_colors)

    # o3d.visualization.draw_geometries([coord_frame, surf_pcd, free_pcd])
    o3d.visualization.draw_geometries([coord_frame, farthest_pcd])
