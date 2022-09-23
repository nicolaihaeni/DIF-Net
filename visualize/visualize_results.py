import os
import h5py
import trimesh
import numpy as np
import open3d as o3d

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points


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


def normalize_pts(pts):
    # pts [N,3]
    center = np.mean(pts, 0)
    pts -= center
    dist = np.linalg.norm(pts, axis=1)
    pts /= np.max(dist * 2)  # align in a sphere with diameter equal to 1
    return pts


def axis_align(pts):
    U, S, Vt = svd_sign_flip(pts)
    return pts @ Vt.T


def svd_sign_flip(X):
    """
    SVD with corrected signs

    Bro, R., Acar, E., & Kolda, T. G. (2008). Resolving the sign ambiguity in the singular value decomposition.
    Journal of Chemometrics: A Journal of the Chemometrics Society, 22(2), 135-140.
    URL: https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf
    """
    # SDV dimensions:
    # U, S, V = np.linalg.svd(X, full_matrices=False)
    # X = U @ diag(S) @ V
    # (I,J) = (I,K) @ (K,K) @ (K,J)

    U, S, V = np.linalg.svd(X, full_matrices=False)

    I = U.shape[0]
    J = V.shape[1]
    K = S.shape[0]

    assert U.shape == (I, K)
    assert V.shape == (K, J)
    assert X.shape == (I, J)

    s = {"left": np.zeros(K), "right": np.zeros(K)}

    for k in range(K):
        mask = np.ones(K).astype(bool)
        mask[k] = False
        # (I,J) = (I,K-1) @ (K-1,K-1) @ (K-1,J)
        Y = X - (U[:, mask] @ np.diag(S[mask]) @ V[mask, :])

        for j in range(J):
            d = np.dot(U[:, k], Y[:, j])
            s["left"][k] += np.sum(np.sign(d) * d**2)
        for i in range(I):
            d = np.dot(V[k, :], Y[i, :])
            s["right"][k] += np.sum(np.sign(d) * d**2)

    for k in range(K):
        if (s["left"][k] * s["right"][k]) < 0:
            if np.abs(s["left"][k]) < np.abs(s["right"][k]):
                s["left"][k] = -s["left"][k]
            else:
                s["right"][k] = -s["right"][k]
        U[:, k] = U[:, k] * np.sign(s["left"][k])
        V[k, :] = V[k, :] * np.sign(s["right"][k])

    return U, S, V


pred_path = "/home/nicolai/phd/code/DIF-Net/logs/plane_eval/recon/test/equi_pose/dfdcc024d1043c73d5dc0e7cb9b4e7b6.ply"
tars_path = "/home/nicolai/phd/code/DIF-Net/logs/plane_eval/recon/test/TARS/TARS_95_dfdcc024d1043c73d5dc0e7cb9b4e7b6_11_pts.npy"
sdf_srn_path = "/home/nicolai/phd/code/DIF-Net/logs/plane_eval/recon/test/SDF_SRN/SRN-SDF_95_dfdcc024d1043c73d5dc0e7cb9b4e7b6_95_11_pts.npy"
# gt_path = "/home/nicolai/phd/code/DIF-Net/logs/plane_eval/recon/test/equi_pose/dfdcc024d1043c73d5dc0e7cb9b4e7b6_gt.ply"
gt_path = "./dfdcc024d1043c73d5dc0e7cb9b4e7b6.h5"

# gt_pts = np.array(trimesh.load(gt_path).vertices)
with h5py.File(gt_path, "r") as hf:
    gt_pts = hf["surface_pts"][:, :3]

mesh = trimesh.load(pred_path)
recon_pts = np.array(trimesh.sample.sample_surface(mesh, 100000)[0])

# downsample pts
pts, _ = sample_farthest_points(torch.Tensor(gt_pts).unsqueeze(0).cuda(), K=10000)
gt_pts = pts.detach().cpu().numpy()[0]

pts, _ = sample_farthest_points(torch.Tensor(recon_pts).unsqueeze(0).cuda(), K=10000)
recon_pts = pts.detach().cpu().numpy()[0]

# Normalize points
gt_pts = normalize_pts(gt_pts)
gt_pts = axis_align(gt_pts)
recon_pts = normalize_pts(recon_pts)
recon_pts = axis_align(recon_pts)

gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
gt_pcd.paint_uniform_color([1, 0, 0])

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(recon_pts))
pcd.translate(np.array([0.0, 1.0, 0.0]))
pcd.paint_uniform_color([0, 1, 0])

axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
o3d.visualization.draw_geometries([pcd, gt_pcd, axis])
