# Copyright (c) Microsoft Corporation.
# Licensed under the MIT Licensepartial_normals.

"""Dataset for DIF-Net.
"""

import h5py
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import Dataset


class Pascal3dDataset(Dataset):
    def __init__(
        self,
        input_file_name,
        on_surface_points,
        expand=-1,
        max_points=200000,
        cam_pose=None,
        symmetry=False,
    ):
        super().__init__()
        self.expand = expand
        with h5py.File(input_file_name, "r") as hf:
            self.partial = hf["points"][:]

        self.max_points = max_points
        self.on_surface_points = on_surface_points

        # If using symmetry, reflect coordinates along symmetry axis
        if symmetry:
            reflected_coords = self.partial.copy()
            reflected_coords[:, 0] = -reflected_coords[:, 0]
            self.partial = np.concatenate([self.partial, reflected_coords])

        # Compute normals and orient towards camera
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.partial))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

        # Now transform the point cloud with the estimated transformation
        trafo = cam_pose
        pcd.transform(trafo)

        self.partial = np.array(pcd.points)
        self.normals = np.array(pcd.normals)
        self.len_coords = self.partial.shape[0]

    def __len__(self):
        if self.max_points != -1:
            return self.max_points // self.on_surface_points
        return self.len_coords // self.on_surface_points

    def __getitem__(self, idx):
        coords = self.partial[:, :3]
        normals = self.normals[:, :3]
        point_cloud_size = coords.shape[0]

        off_surface_samples = self.on_surface_points
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        if self.expand != -1:
            on_surface_coords += (
                on_surface_normals * self.expand
            )  # expand the shape surface if its structure is too thin

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        ground_truth = {
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }
        observation = {
            "coords": torch.from_numpy(coords).float(),
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }
        return observation, ground_truth


if __name__ == "__main__":

    def rotate_pascal3d_to_shapenet():
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        rot_y = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return rot_x @ rot_y

    cam_poses = np.load(
        "/home/nicolai/phd/data/test_data/pascal3d_car/pascal3d_car_equi_pose.npy",
        allow_pickle=True,
    ).item()

    # Get the right camera pose
    basename = "02814533_10251"
    index = cam_poses["names"].index(basename)
    cam_pose = cam_poses["est_w_T_cam"][index]
    rot_x = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pred_cam_pose = rot_x @ rotate_pascal3d_to_shapenet() @ cam_pose

    with h5py.File(
        "/home/nicolai/phd/data/test_data/pascal3d_car/pascal3d_imagenet_car_val.h5",
        "r",
    ) as hf:
        names = hf["Names"][:].tolist()
        names = [f[0].decode()[1:] for f in names]
        index = names.index(basename)
        gt_points = hf["points"][index][:, :3]
        gt_cam_pose = hf["poses"][index]

    sdf_dataset = Pascal3dDataset(
        input_file_name="/home/nicolai/phd/data/test_data/pascal3d_car/partials/02814533_10251_partial_pcd.h5",
        cam_pose=pred_cam_pose,
        on_surface_points=4000,
        symmetry=False,
    )

    # Recreate the point clouds and check if they align
    surface_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
    surface_pcd.paint_uniform_color(np.array([1, 0, 0]))
    surface_pcd.transform(rotate_pascal3d_to_shapenet())

    partial_points = sdf_dataset.partial
    partial_normals = sdf_dataset.normals
    partial_est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_points))
    partial_est.normals = o3d.utility.Vector3dVector(partial_normals)
    partial_est.paint_uniform_color(np.array([0, 0, 1]))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([surface_pcd, partial_est, axis])
