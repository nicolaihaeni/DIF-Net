# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Chamfer distance calculation.
"""

import os
import h5py
import trimesh
import numpy as np

import torch
import open3d as o3d
from pytorch3d.loss import chamfer_distance


def normalize_pts(pts):
    # pts [N,3]
    center = np.mean(pts, 0)
    pts -= center
    dist = np.linalg.norm(pts, axis=1)
    pts /= np.max(dist * 2)  # align in a sphere with diameter equal to 1
    return pts


def compute_chamfer(recon_pts, gt_pts):
    with torch.no_grad():
        recon_pts = torch.from_numpy(recon_pts).float().cuda()[None, ...]
        gt_pts = torch.from_numpy(gt_pts).float().cuda()[None, ...]
        dist, _ = chamfer_distance(recon_pts, gt_pts, batch_reduction=None)
        dist = dist.detach().cpu().squeeze().numpy()
    return dist


def compute_f1(recon_pts, gt_pts, th=0.01):
    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    recon_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(recon_pts))
    d1 = gt_pcd.compute_point_cloud_distance(recon_pcd)
    d2 = recon_pcd.compute_point_cloud_distance(gt_pcd)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


def compute_recon_error(recon_path, gt_points):
    recon_mesh = trimesh.load(recon_path)
    if isinstance(recon_mesh, trimesh.Scene):
        recon_mesh = recon_mesh.dump().sum()

    recon_pts = np.array(trimesh.sample.sample_surface(recon_mesh, 100000)[0])

    # Normalize points
    recon_pts = normalize_pts(recon_pts)
    gt_pts = normalize_pts(gt_pts)

    # Choose random subset
    np.random.seed(0)
    idx = np.random.choice(len(recon_pts), size=(num_pts), replace=True)
    recon_pts = recon_pts[idx, :]

    idx = np.random.choice(len(gt_pts), size=(num_pts), replace=True)
    gt_pts = gt_pts[idx, :]

    cd = compute_chamfer(recon_pts, gt_pts)
    f1, _, _ = compute_f1(recon_pts, gt_pts)
    return cd, f1
