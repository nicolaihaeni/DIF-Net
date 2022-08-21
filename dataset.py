# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dataset for DIF-Net.
"""

import os
import json
import h5py
import numpy as np

import torch
import open3d as o3d
from torch.utils.data import Dataset
from dgl.geometry import farthest_point_sampler


class PointCloudDataset(Dataset):
    def __init__(self, instance_idx, instance_path, on_surface_points):
        super().__init__()

        self.on_surface_points = on_surface_points
        self.instance_idx = instance_idx

        print(f"Loading point cloud of subject {self.instance_idx}")
        with h5py.File(instance_path, "r") as hf:
            free_points = hf["free_pts"][:]
            surface_points = hf["surface_pts"][:]

        self.coords, self.normals, self.sdf = np.split(surface_points, [3, 6], axis=-1)
        self.free_points_coords, self.free_points_sdf = np.split(
            free_points, [3], axis=-1
        )

        # Voxel downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords)
        down_pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Conditional input farthest point sampling
        points = np.asarray(down_pcd.points)
        farthest_points, _ = farthest_point_sampler(
            torch.tensor(points).unsqueeze(0), K=1024
        )
        self.farthest_points = farthest_points.squeeze(0).numpy()

    def __len__(self):
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

        free_rand_idcs = np.random.choice(free_point_size, size=off_surface_samples)
        free_points_coords = self.free_points_coords[free_rand_idcs, :]
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        # if a free space point has gt SDF value, replace -1 with it.
        sdf[self.on_surface_points :, :] = self.free_points_sdf[free_rand_idcs]

        coords = np.concatenate((on_surface_coords, free_points_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {
            "coords": torch.from_numpy(coords).float(),
            "farthest_points": torch.from_numpy(self.farthest_points)
            .permute(1, 0)
            .float(),
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }


class PointCloudSingleDataset(Dataset):
    def __init__(self, root_dir, split_file, on_surface_points, train=False):
        super().__init__()

        self.on_surface_points = on_surface_points
        self.root_dir = root_dir
        print(root_dir)

        with open(split_file, "r") as in_file:
            data = json.load(in_file)

        if train:
            split_data = data["train"]
        else:
            split_data = data["test"]

        self.instances = []
        for cat in split_data:
            for filename in split_data[cat]:
                self.instances.append(
                    os.path.join(root_dir, cat, filename, "ori_sample.h5")
                )

        assert len(self.instances) != 0, "No objects in the data directory"
        self.num_instances = len(self.instances)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        with h5py.File(self.instances[idx]) as hf:
            free_points = hf["free_pts"][:]
            surface_points = hf["surface_pts"][:]

        coords, normals, sdf = np.split(surface_points, [3, 6], axis=-1)
        free_points_coords, free_points_sdf = np.split(free_points, [3], axis=-1)

        # Voxel downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        down_pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Conditional input farthest point sampling
        points = np.asarray(down_pcd.points)
        farthest_points = farthest_point_sampler(
            pos=torch.tensor(points).unsqueeze(0), npoints=1024
        )
        farthest_points = points[farthest_points.squeeze(0).numpy()]
        farthest_points = np.transpose(farthest_points)

        point_cloud_size = coords.shape[0]
        free_point_size = free_points_coords.shape[0]

        off_surface_samples = self.on_surface_points
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = coords[rand_idcs, :]
        on_surface_normals = normals[rand_idcs, :]

        free_rand_idcs = np.random.choice(free_point_size, size=off_surface_samples)
        free_points_coords = free_points_coords[free_rand_idcs, :]
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        # if a free space point has gt SDF value, replace -1 with it.
        sdf[self.on_surface_points :, :] = free_points_sdf[free_rand_idcs]

        coords = np.concatenate((on_surface_coords, free_points_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        gt = {
            "normals": torch.from_numpy(normals).float(),
            "sdf": torch.from_numpy(sdf).float(),
        }
        model_inputs = {
            "coords": torch.from_numpy(coords).float(),
            "farthest_points": torch.from_numpy(farthest_points).float(),
        }
        return model_inputs, gt


class PointCloudMultiDataset(Dataset):
    def __init__(self, root_dir, split_file, on_surface_points, train=False):
        # This class adapted from SIREN https://vsitzmann.github.io/siren/
        super().__init__()

        self.root_dir = root_dir
        print(root_dir)

        with open(split_file, "r") as in_file:
            data = json.load(in_file)

        if train:
            split_data = data["train"]
        else:
            split_data = data["test"]

        self.instances = []
        for cat in split_data:
            for filename in split_data[cat]:
                self.instances.append(
                    os.path.join(root_dir, cat, filename, "ori_sample.h5")
                )

        assert len(self.instances) != 0, "No objects in the data directory"

        # Load all the data into memory
        self.all_instances = [
            PointCloudDataset(
                idx,
                instance_path,
                on_surface_points=on_surface_points,
            )
            for idx, instance_path in enumerate(self.instances)
        ]
        self.num_instances = len(self.all_instances)

    def __len__(self):
        return self.num_instances

    def __getitem__(self, idx):
        return self.all_instances[idx]
