import os
import numpy as np
import open3d as o3d
import h5py

import torch
from dgl.geometry import farthest_point_sampler


def sample_spherical(n, radius=1.0):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    return np.transpose(xyz)


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


# All the following functions follow the opencv convention for camera coordinates.
def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0.0, -1.0, 0.0])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


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

    # Compute non visible points
    cam_radius = 2.0
    cam = sample_spherical(1, cam_radius)
    radius = 500 * cam_radius
    _, pt_map = pcd.hidden_point_removal(cam, radius)
    visible_pcd = pcd.select_by_index(pt_map)
    visible_points = np.asarray(visible_pcd.points)
    visible_pcd = visible_pcd.translate((1.0, 0.0, 0.0))
    if VISUALIZE:
        o3d.visualization.draw_geometries([visible_pcd])

    # # Downsample and do farthest point sampling
    # down_sampled_pcd = o3d.geometry.PointCloud()
    # down_sampled_pcd.points = o3d.utility.Vector3dVector(visible_points)
    # colors = np.zeros_like(visible_points)
    # colors[:, 2] = 1
    # down_sampled_pcd.colors = o3d.utility.Vector3dVector(colors)
    # down_sampled_pcd = down_sampled_pcd.voxel_down_sample(voxel_size=0.01)
    # down_sampled_pcd = down_sampled_pcd.translate((1.0, 0.0, 0.0))

    # Do farthest point sampling
    # down_sampled_points = np.asarray(down_sampled_pcd.points)
    print(visible_points.shape[0])
    if visible_points.shape[0] <= 1024:
        print("Not enough points!")
    farthest_points = farthest_point_sampler(
        pos=torch.tensor(visible_points).unsqueeze(0), npoints=1024
    )
    farthest_points = visible_points[farthest_points.squeeze(0).numpy()]
    farthest_pcd = o3d.geometry.PointCloud()
    farthest_pcd.points = o3d.utility.Vector3dVector(farthest_points)
    colors = np.zeros_like(farthest_points)
    colors[:, 1] = 1
    farthest_pcd.colors = o3d.utility.Vector3dVector(colors)
    farthest_pcd = farthest_pcd.translate((1.0, 0.0, 0.0))

    trafo = look_at(cam.squeeze(), np.array([0, 0, 0]))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis.transform(trafo)
    o3d.visualization.draw_geometries([pcd, farthest_pcd, visible_pcd, axis])
