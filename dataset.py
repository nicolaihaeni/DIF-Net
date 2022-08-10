# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dataset for DIF-Net.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from pytorch3d.ops import sample_farthest_points


class PointCloudDataset(Dataset):
    def __init__(
        self,
        pointcloud_path,
        on_surface_points,
        instance_idx=None,
        expand=-1,
        max_points=-1,
    ):
        super().__init__()

        self.instance_idx = instance_idx
        self.expand = expand

        print("Loading point cloud of subject%04d" % self.instance_idx)
        # surface points
        point_cloud = loadmat(pointcloud_path)
        point_cloud = point_cloud["p"]

        # surface points with normals
        self.coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        # Subsample the point cloud to get pointnet inputs
        points, idx = sample_farthest_points(
            torch.tensor(self.coords).unsqueeze(0).cuda(),
            K=1024,
        )
        self.farthest_coords = points.squeeze(0).cpu().numpy()

        # free space points
        free_points = loadmat(
            pointcloud_path.replace("surface_pts_n_normal", "free_space_pts")
        )
        free_points = free_points["p_sdf"]
        print("Finished loading point cloud")

        free_points_coords = free_points[:, :3]
        free_points_sdf = free_points[:, 3:]

        self.free_points_coords = free_points_coords
        self.free_points_sdf = free_points_sdf

        self.on_surface_points = on_surface_points
        self.max_points = max_points

    def __len__(self):
        if self.max_points != -1:
            return self.max_points // self.on_surface_points
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]
        free_point_size = self.free_points_coords.shape[0]

        off_surface_samples = self.on_surface_points
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        if self.expand != -1:
            on_surface_coords += (
                on_surface_normals * self.expand
            )  # expand the shape surface if its structure is too thin

        off_surface_coords = np.random.uniform(
            -1, 1, size=(off_surface_samples // 2, 3)
        )
        free_rand_idcs = np.random.choice(
            free_point_size, size=off_surface_samples // 2
        )
        free_points_coords = self.free_points_coords[free_rand_idcs, :]

        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        # if a free space point has gt SDF value, replace -1 with it.
        if self.expand != -1:
            sdf[self.on_surface_points + off_surface_samples // 2 :, :] = (
                self.free_points_sdf[free_rand_idcs] - self.expand
            )
        else:
            sdf[
                self.on_surface_points + off_surface_samples // 2 :, :
            ] = self.free_points_sdf[free_rand_idcs]

        coords = np.concatenate(
            (on_surface_coords, off_surface_coords, free_points_coords), axis=0
        )
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {
            "coords": torch.from_numpy(coords).float(),
            "farthest_coords": torch.from_numpy(self.farthest_coords)
            .permute(1, 0)
            .float(),
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze().long(),
        }
