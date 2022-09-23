# Copyright (c) Microsoft Corporation.
# Licensed under the MIT Licensepartial_normals.

"""Dataset for DIF-Net.
"""

import h5py
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import Dataset

import utils


class ShapenetDataset(Dataset):
    def __init__(
        self,
        file_name,
        on_surface_points,
        instance_idx=None,
        expand=-1,
        max_points=200000,
    ):
        super().__init__()

        self.instance_idx = instance_idx
        self.on_surface_points = on_surface_points
        self.file_name = file_name
        self.max_points = max_points
        self.expand = expand
        self.len_coords = 500000

        print(f"Loading data of subject {self.instance_idx}.")
        with h5py.File(self.file_name, "r") as hf:
            self.coords = hf["surface_pts"][:]
            self.free_points = hf["free_pts"][:]

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

        return {
            "coords": torch.from_numpy(coords).float(),
            "sdf": torch.from_numpy(sdf).float(),
            "normals": torch.from_numpy(normals).float(),
            "instance_idx": torch.tensor([self.instance_idx]).squeeze().long(),
        }


class ShapenetEvalDataset(Dataset):
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
            self.partial = hf["pcd"][:]

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
        roty = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        trafo = roty @ cam_pose
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


class ShapenetMultiDataset(Dataset):
    def __init__(
        self,
        file_names,
        on_surface_points,
        max_points=-1,
        expand=-1,
        **kwargs,
    ):
        self.on_surface_points = on_surface_points
        self.instances = file_names
        assert len(self.instances) != 0, "No objects in the data directory"

        self.all_instances = [
            ShapenetDataset(
                file_name=dir,
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


if __name__ == "__main__":
    cam_poses = np.load(
        "/home/nicolai/phd/data/test_data/shapenet_plane_test/shapene_plane_pose.npy",
        allow_pickle=True,
    ).item()

    # Get the right camera pose
    basename = "dfdcc024d1043c73d5dc0e7cb9b4e7b6"
    index = cam_poses["names"].index(basename)
    gt_cam_pose = cam_poses["gt_w_T_cam"][index]
    pred_cam_pose = cam_poses["est_w_T_cam"][index]

    sdf_dataset = PointCloudEvalDataset(
        ground_truth_file_name="./example_dir/dfdcc024d1043c73d5dc0e7cb9b4e7b6/dfdcc024d1043c73d5dc0e7cb9b4e7b6.h5",
        input_file_name="/home/nicolai/phd/data/test_data/shapenet_plane_test/dfdcc024d1043c73d5dc0e7cb9b4e7b6.h5",
        cam_pose=pred_cam_pose,
        on_surface_points=4000,
        symmetry=False,
    )

    gt_points = sdf_dataset.coords[:, :3]
    partial_points = sdf_dataset.partial
    partial_normals = sdf_dataset.normals

    # Recreate the point clouds and check if they align
    import open3d as o3d

    surface_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
    surface_pcd.paint_uniform_color(np.array([1, 0, 0]))

    partial_est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_points))
    partial_est.normals = o3d.utility.Vector3dVector(partial_normals)
    partial_est.paint_uniform_color(np.array([0, 0, 1]))

    print(
        np.array(partial_est.points).min(axis=0),
        np.array(partial_est.points).max(axis=0),
    )

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([surface_pcd, partial_est, axis])
