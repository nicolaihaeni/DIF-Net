import os

import trimesh
import numpy as np
import open3d as o3d


mesh_path = "/home/nicolai/Downloads/e557af9d5aa40f424d210d9468aedaf2/models/model_normalized.obj"

mesh = trimesh.load(mesh_path, force="mesh")

points = np.array(trimesh.sample.sample_surface(mesh, 10000)[0])

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3)
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

_, pt_map = pcd.hidden_point_removal(np.array([-1, -1, 0]), radius=100)
pcd = pcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([pcd])
