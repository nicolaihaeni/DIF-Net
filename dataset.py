# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dataset for DIF-Net.
"""

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from pytorch3d.ops import sample_farthest_points


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, split_file, on_surface_points, train=False):
        super().__init__()

        self.instance_idx = instance_idx

        with open(split_file, "r") as in_file:
            data = json.load(in_file)
        if train:
            split_data = data["train"]
        else:
            split_data = data["test"]

        self.filenames = []
        for cat in split_data:
            for name in split_data[cat]:
                self.filenames.append(os.path.join(src_dir, cat, name, "ori_sample.h5"))

        self.on_surface_points = on_surface_points

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        h5file = h5py.File(self.filenames[idx], "r")
        free_points = h5file["free_pts"][:]
        surface_points = h5file["surface_pts"][:]

        coords, normals = np.split(surface_points, [3, 6], axis=-1)
        free_points_coords, free_points_sdf = np.split(free_points_sdf, [3, 6], axis=-1)

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

        # Conditional input farthest point sampling
        points, idx = sample_farthest_points(
            torch.tensor(self.coords).unsqueeze(0).cuda(),
            K=1024,
        )
        farthest_points = points.squeeze(0).cpu().numpy()

        return {
            "coords": torch.from_numpy(coords).float(),
            "farthest_points": torch.from_numpy(self.farthest_points)
            .permute(1, 0)
            .float(),
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
        }
