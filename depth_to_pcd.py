# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Training script for DIF-Net.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import yaml
import h5py
import numpy as np
import dataset, utils, training_loop, loss, modules, meta_modules

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from unet import DepthPredictor, NormalNet
import imageio


def depth_2_normal(depth, depth_unvalid, K):
    H, W = depth.shape
    grad_out = np.zeros((H, W, 3))
    X, Y = np.meshgrid(np.arange(0, W), np.arange(0, H))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = ((X - cx) / fx) * depth
    Y = ((Y - cy) / fy) * depth

    XYZ_camera = np.stack([X, Y, depth], axis=-1)

    # compute tangent vectors
    vx = XYZ_camera[1:-1, 2:, :] - XYZ_camera[1:-1, 1:-1, :]
    vy = XYZ_camera[2:, 1:-1, :] - XYZ_camera[1:-1, 1:-1, :]

    # finally compute cross product
    normal = np.cross(vx.reshape(-1, 3), vy.reshape(-1, 3))
    normal_norm = np.linalg.norm(normal, axis=-1)
    normal = np.divide(normal, normal_norm[:, None])

    # reshape to image
    normal_out = normal.reshape(H - 2, W - 2, 3)
    grad_out[1:-1, 1:-1, :] = 0.5 - 0.5 * normal_out

    # zero out +Inf
    grad_out[depth_unvalid] = 0.0
    return grad_out


if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    p.add_argument("--config", type=str, default="", help="training configuration.")
    p.add_argument("--root_dir", type=str, default="", help="training data path.")
    p.add_argument("--filename", type=str, default="", help="training subject names.")

    p.add_argument(
        "--logging_root", type=str, default="./logs", help="root for logging"
    )
    p.add_argument(
        "--experiment_name",
        type=str,
        default="default",
        help="Name of subdirectory in logging_root where summaries and checkpoints will be saved.",
    )

    # General training options
    p.add_argument("--batch_size", type=int, default=256, help="training batch size.")
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate. default=1e-4")
    p.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs to train for."
    )

    # load configs if exist
    opt = p.parse_args()
    if opt.config == "":
        meta_params = vars(opt)
    else:
        with open(opt.config, "r") as stream:
            meta_params = yaml.safe_load(stream)

    # define dataloader
    if ".h5" in opt.filename:
        with h5py.File(opt.filename, "r") as hf:
            imgs = hf["rgb"][1] / 255.0
            masks = hf["mask"][1]
            gt_depth = hf["depth"][1]
    elif ".jpg" or ".png" in opt.filename:
        imgs = imageio.imread(opt.filename)[None]
        masks = imageio.imread(opt.filename)[None]

    # create save path
    root_path = os.path.join(
        meta_params["logging_root"], meta_params["experiment_name"]
    )
    utils.cond_mkdir(root_path)

    with io.open(os.path.join(root_path, "model.yml"), "w", encoding="utf8") as outfile:
        yaml.dump(meta_params, outfile, default_flow_style=False, allow_unicode=True)

    # define DIF-Net
    normal_net = NormalNet().cuda()
    _, normal_net, _ = utils.load_checkpoints(meta_params, normal_net, subdir="normals")
    depth_net = DepthPredictor().cuda()
    _, depth_net, _ = utils.load_checkpoints(meta_params, depth_net)

    # Run the images through the depth prediction network
    normal = normal_net(
        torch.tensor(imgs).unsqueeze(0).cuda(), torch.tensor(masks).unsqueeze(0).cuda()
    )
    depth = depth_net(
        torch.tensor(imgs).unsqueeze(0).cuda(),
        torch.tensor(masks).unsqueeze(0).cuda(),
        normal,
    )

    import matplotlib.pyplot as plt
    import open3d as o3d

    depth = depth[0].detach().cpu().numpy()
    depth[masks == 0] = 0.0
    gt_depth[masks == 0] = 0.0

    depth_unvalid = masks.astype(bool)
    depth_unvalid = ~depth_unvalid
    normal = depth_2_normal(
        depth,
        depth_unvalid,
        np.array([[262.5, 0.0, 128.0], [0.0, 262.5, 128.0], [0.0, 0.0, 1.0]]),
    )
    normal_gt = depth_2_normal(
        gt_depth,
        depth_unvalid,
        np.array([[262.5, 0.0, 128.0], [0.0, 262.5, 128.0], [0.0, 0.0, 1.0]]),
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(normal)
    ax[1].imshow(normal_gt)
    plt.show()

    u, v = np.where(depth != 0.0)
    y = depth[u, v] * ((u - 128.0) / 262.5)
    x = depth[u, v] * ((v - 128.0) / 262.5)
    z = depth[u, v]
    pts = np.stack([x, y, z], axis=-1)

    pcd_part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd_part.paint_uniform_color(np.array([0, 1, 0]))

    u, v = np.where(gt_depth != 0.0)
    y = gt_depth[u, v] * ((u - 128.0) / 262.5)
    x = gt_depth[u, v] * ((v - 128.0) / 262.5)
    z = gt_depth[u, v]
    pts = np.stack([x, y, z], axis=-1)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color(np.array([1, 0, 0]))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, axis, pcd_part])
