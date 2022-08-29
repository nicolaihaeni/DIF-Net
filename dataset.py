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
    def __init__(
        self,
        point_cloud_path,
        on_surface_points,
        instance_idx=None,
        expand=-1,
        max_points=200000,
    ):
        super().__init__()

        self.instance_idx = instance_idx
        self.on_surface_points = on_surface_points
        self.point_cloud_path = point_cloud_path
        self.max_points = max_points
        self.expand = expand

        print(f"Loading data of subject {self.instance_idx}")
        with h5py.File(point_cloud_path, "r") as hf:
            coords = hf["surface_pts"][:]
            free_points = hf["free_pts"][:]

        self.coords = coords[:, :3]
        self.normals = coords[:, 3:6]
        self.free_points_coords = free_points[:, :3]
        self.free_points_sdf = free_points[:, 3]

        # Voxel downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords)
        down_pcd = pcd.voxel_down_sample(voxel_size=0.01)

        # Conditional input farthest point sampling
        points = np.asarray(down_pcd.points)
        farthest_points = farthest_point_sampler(
            pos=torch.tensor(points).unsqueeze(0), npoints=1024
        )
        self.farthest_points = np.transpose(points[farthest_points.squeeze(0).numpy()])

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
                self.free_points_sdf[free_rand_idcs][:, None] - self.expand
            )
        else:
            sdf[
                self.on_surface_points + off_surface_samples // 2 :, :
            ] = self.free_points_sdf[free_rand_idcs][:, None]

        coords = np.concatenate(
            (on_surface_coords, off_surface_coords, free_points_coords), axis=0
        )
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {
            "coords": torch.from_numpy(coords).float(),
            "sdf": torch.from_numpy(sdf).float(),
            "farthest_points": torch.from_numpy(self.farthest_points).float(),
            "normals": torch.from_numpy(normals).float(),
            "instance_idx": torch.tensor([self.instance_idx]).squeeze().long(),
        }


class PointCloudMultiDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split_file,
        on_surface_points,
        max_points=-1,
        expand=-1,
        train=False,
    ):
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

        self.all_instances = [
            PointCloudDataset(
                point_cloud_path=dir,
                on_surface_points=on_surface_points,
                max_points=max_points,
                instance_idx=idx,
                expand=expand,
            )
            for idx, dir in enumerate(self.instances)
        ]

        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend([bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                    ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [
            {"sdf": obj["sdf"], "normals": obj["normals"]} for obj in observations
        ]

        return observations, ground_truth
