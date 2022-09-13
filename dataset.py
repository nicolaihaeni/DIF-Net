# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dataset for DIF-Net.
"""

import os
import json
import h5py
import numpy as np
import open3d as o3d

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from dgl.geometry import farthest_point_sampler

import utils


class PointCloudDataset(Dataset):
    def __init__(
        self,
        file_name,
        on_surface_points,
        instance_idx=None,
        expand=-1,
        max_points=200000,
        cam_pose=None,
        symmetry=False,
    ):
        super().__init__()

        self.instance_idx = instance_idx
        self.on_surface_points = on_surface_points
        self.file_name = file_name
        self.max_points = max_points
        self.expand = expand
        self.len_coords = 500000
        self.is_test = cam_pose is not None

        print(f"Loading data of subject {self.instance_idx}.")
        with h5py.File(self.file_name, "r") as hf:
            self.coords = hf["surface_pts"][:]
            self.free_points = hf["free_pts"][:]

        if cam_pose is not None:
            self.gt_coords = self.coords[:, :3]
            pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self.coords[:, :3])
            )
            _, pt_map = pcd.hidden_point_removal(cam_pose, 6.0 * 100)
            self.coords = self.coords[pt_map, :]

            if symmetry:
                reflected_coords = self.coords.copy()
                reflected_coords[:, 0] = -reflected_coords[:, 0]
                self.coords = np.concatenate([self.coords, reflected_coords])

    def __len__(self):
        if self.max_points != -1:
            return self.max_points // self.on_surface_points
        return self.len_coords // self.on_surface_points

    def __getitem__(self, idx):
        coords = self.coords[:, :3]
        normals = self.coords[:, 3:6]
        free_points_coords = self.free_points[:, :3]
        free_points_sdf = self.free_points[:, 3]

        point_cloud_size = coords.shape[0]
        free_point_size = free_points_coords.shape[0]

        off_surface_samples = self.on_surface_points
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = coords[rand_idcs, :]
        on_surface_normals = normals[rand_idcs, :]

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
        free_points_coords = free_points_coords[free_rand_idcs, :]

        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points :, :] = -1  # off-surface = -1

        # if a free space point has gt SDF value, replace -1 with it.
        if self.expand != -1:
            sdf[self.on_surface_points + off_surface_samples // 2 :, :] = (
                free_points_sdf[free_rand_idcs][:, None] - self.expand
            )
        else:
            sdf[
                self.on_surface_points + off_surface_samples // 2 :, :
            ] = free_points_sdf[free_rand_idcs][:, None]

        coords = np.concatenate(
            (on_surface_coords, off_surface_coords, free_points_coords), axis=0
        )
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        if self.is_test:
            num_partial_points = self.on_surface_points + off_surface_samples // 2
            return {
                "coords": torch.from_numpy(coords[:num_partial_points]).float(),
                "sdf": torch.from_numpy(sdf[:num_partial_points]).float(),
                "normals": torch.from_numpy(normals[:num_partial_points]).float(),
                "instance_idx": torch.tensor([self.instance_idx]).squeeze().long(),
            }

        return {
            "coords": torch.from_numpy(coords).float(),
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
            "instance_idx": torch.tensor([self.instance_idx]).squeeze().long(),
        }


class PointCloudMultiDataset(Dataset):
    def __init__(
        self,
        file_names,
        on_surface_points,
        max_points=-1,
        expand=-1,
        cam_pose=None,
        symmetry=False,
        **kwargs,
    ):
        self.on_surface_points = on_surface_points

        self.instances = file_names
        assert len(self.instances) != 0, "No objects in the data directory"

        self.all_instances = [
            PointCloudDataset(
                file_name=dir,
                on_surface_points=on_surface_points,
                max_points=max_points,
                instance_idx=idx,
                expand=expand,
                cam_pose=cam_pose,
                symmetry=symmetry,
            )
            for idx, dir in enumerate(self.instances)
        ]

        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_point_clouds(self, idx):
        return (
            self.all_instances[idx].coords[:, :3],
            self.all_instances[idx].gt_coords[:, :3],
        )

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


class DepthDataset(Dataset):
    def __init__(self, file_paths, num_views=30, train=False):
        super().__init__()

        self.file_paths = file_paths
        self.num_views = num_views
        self.length = len(file_paths) * num_views
        self.is_train = train

        self.transform_list = [
            "identity",
            "jitter",
            "gray",
            "equalize",
            "posterize",
            "solarize",
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx // self.num_views

        view_idx = np.random.randint(0, self.num_views)
        with h5py.File(self.file_paths[idx], "r") as hf:
            image = hf["rgb"][view_idx] / 255.0
            depth = hf["depth"][view_idx]
            mask = hf["mask"][view_idx]
            K = hf["K"][:]
            cam2world = hf["cam2world"][view_idx]

        depth[depth == 100.0] = 10.0

        if self.is_train:
            # Bounding box computation and move bbox
            x, y = np.where(mask)
            bbox = min(x), max(x), min(y), max(y)
            W, H = 256, 256
            w, h = bbox[1] - bbox[0], bbox[3] - bbox[2]

            prob = np.random.rand()
            if prob > 0.5:
                min_range = min(1.0, 40 / (min(h, w)))
                max_range = max(min_range, (256 * 0.9) / (max(h, w)))

                scale_factor = np.random.uniform(min_range, max_range)
                scale_factor = int(scale_factor * W) / W
                image = utils.resize_array(image, bbox, scale_factor, pad_value=1.0)
                depth = utils.resize_array(
                    depth, bbox, scale_factor, pad_value=10.0, inter="nearest"
                )
                mask = utils.resize_array(
                    mask, bbox, scale_factor, pad_value=0.0, inter="nearest"
                )
                # depth = depth / scale_factor
                depth[np.where(mask == 0)] = 10.0
                image[np.where(mask == 0)] = 1.0

        image = T.ToTensor()(image)
        depth = torch.tensor(depth).float()
        mask = torch.tensor(mask).float()

        # Image augmentations
        if self.is_train:
            probs = torch.ones(len(self.transform_list))
            dist = torch.distributions.categorical.Categorical(probs=probs)
            aug = self.transform_list[dist.sample()]

            if aug == "jitter":
                image = T.ColorJitter(
                    brightness=(0.25, 0.75), hue=(-0.4, 0.4), saturation=(0.25, 0.75)
                )(image)
            if aug == "gray":
                image = T.Grayscale(3)(image)
            if aug == "equalize":
                image = (image * 255.0).to(dtype=torch.uint8)
                image = T.RandomEqualize(1.0)(image)
                image = (image / 255.0).float()
            if aug == "posterize":
                image = (image * 255.0).to(dtype=torch.uint8)
                image = T.RandomPosterize(bits=2, p=1.0)(image)
                image = (image / 255.0).float()
            if aug == "solarize":
                image = (image * 255.0).to(dtype=torch.uint8)
                image = T.RandomSolarize(threshold=192, p=1.0)(image)
                image = (image / 255.0).float()

        image = image.permute(1, 2, 0).clone()
        image[torch.where(mask == 0)] = 1.0

        return {
            "images": image.numpy(),
            "depths": depth.numpy(),
            "masks": mask.numpy(),
            "K": torch.tensor(K).float(),
            "cam2world": torch.tensor(cam2world).float().numpy(),
        }


if __name__ == "__main__":
    point_filename = "example_dir/1c14ef4c48b7d95d61c6c61410fc904b/1c14ef4c48b7d95d61c6c61410fc904b.h5"
    filename = "example_dir/1c14ef4c48b7d95d61c6c61410fc904b/1c14ef4c48b7d95d61c6c61410fc904b_rgbd.h5"
    dataset = DepthDataset(file_paths=[filename])

    view_idx = 1
    with h5py.File(filename, "r") as hf:
        image = hf["rgb"][view_idx] / 255.0
        depth_before = hf["depth"][view_idx]
        mask = hf["mask"][view_idx]

    with h5py.File(point_filename, "r") as hf:
        surface_points = hf["surface_pts"][:, :3]

    for ii in range(50):
        data = dataset[0]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(image)
        ax[0, 1].imshow(depth_before, cmap="plasma", vmin=0, vmax=10.0)
        ax[0, 2].imshow(mask, cmap="gray")

        ax[1, 0].imshow(data["images"])
        ax[1, 1].imshow(data["depths"], cmap="plasma", vmin=0.0, vmax=10.0)
        ax[1, 2].imshow(data["masks"], cmap="gray")
        plt.show()

    # # Recreate the point clouds and check if they align
    # import open3d as o3d

    # surface_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(surface_points))
    # surface_pcd.paint_uniform_color(np.array([1, 0, 0]))

    # depth_before[depth_before == 10.0] = np.inf
    # depth_after = data["depths"]
    # depth_after[depth_after == 10.0] = np.inf

    # u, v = np.where(depth_before != np.inf)
    # y = depth_before[u, v] * ((u - 128.0) / 262.5)
    # x = depth_before[u, v] * ((v - 128.0) / 262.5)
    # z = depth_before[u, v]
    # pts = np.stack([x, y, z], axis=-1)

    # pcd_before = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # pcd_before.paint_uniform_color(np.array([0, 1, 0]))
    # pcd_before.transform(data["cam2world"])
    # pcd_before.paint_uniform_color(np.array([0, 1, 0]))

    # u, v = np.where(depth_after != np.inf)
    # y = depth_after[u, v] * ((u - 128.0) / 262.5)
    # x = depth_after[u, v] * ((v - 128.0) / 262.5)
    # z = depth_after[u, v]
    # pts = np.stack([x, y, z], axis=-1)

    # pcd_after = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # pcd_after.paint_uniform_color(np.array([0, 0, 1]))
    # pcd_after.transform(data["cam2world"])

    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # camera.transform(data["cam2world"])
    # o3d.visualization.draw_geometries(
    # [surface_pcd, pcd_before, pcd_after, axis, camera]
    # )
