# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import numpy as np

import cv2
import torch
import torch.nn as nn


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoints(path, model, optimizer, global_step):
    """Save model, optimzer and global iteration to file"""
    if isinstance(model, torch.nn.DataParallel):
        torch.save(
            {
                "epoch": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
    else:
        torch.save(
            {
                "epoch": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )


def load_checkpoints(args, model, optimizer=None, subdir=None):
    """Load model, optimzer and global iteration from file"""
    start = 0

    # Load checkpoints
    if subdir is not None:
        experiment_path = os.path.join(
            args["logging_root"], args["experiment_name"], "checkpoints", subdir
        )
    else:
        experiment_path = os.path.join(
            args["logging_root"], args["experiment_name"], "checkpoints"
        )

    if not os.path.exists(experiment_path):
        return start, model, optimizer

    ckpts = [
        os.path.join(experiment_path, f)
        for f in sorted(os.listdir(experiment_path))
        if f.endswith(".pth")
    ]

    print(f"Found checkpoints {ckpts}")
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print(f"Reloading checkpoints from {ckpt_path}")
        ckpt = torch.load(ckpt_path)

        start = ckpt["epoch"]
        optimizer = torch.optim.Adam(params=model.parameters()).load_state_dict(
            ckpt["optimizer_state_dict"]
        )

        # Load model
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
    return start, model, optimizer


def resize_array(array, bbox, scale_factor, pad_value, inter=None):
    temp = array[bbox[0] : bbox[1], bbox[2] : bbox[3]]
    if inter is not None:
        temp = cv2.resize(
            temp,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        temp = cv2.resize(
            temp, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
        )

    h, w = temp.shape[:2]
    out_array = np.ones_like(array) * pad_value
    out_array[
        int(128 - h // 2) : int(128 - h // 2) + h,
        int(128 - w // 2) : (128 - w // 2) + w,
    ] = temp
    return out_array


def get_filenames(root_dir, split_file, mode="train", depth=False):
    with open(split_file, "r") as in_file:
        data = json.load(in_file)[mode]

    instances = []
    for cat in data:
        for filename in data[cat]:
            if depth:
                instances.append(
                    os.path.join(root_dir, cat, filename, f"{filename}_rgbd.h5")
                )
            else:
                instances.append(
                    os.path.join(root_dir, cat, filename, f"{filename}.h5")
                )
    return instances


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def sample_spherical(n, radius=3.0):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    return xyz


def to_png(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def depth_to_png(depth):
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    old_min, old_max = 0, 10.0
    new_min, new_max = 0, 255
    depth = (new_max - new_min) / (old_max - old_min) * (depth - old_min) + new_min
    return np.clip(depth, 0, 255).astype(np.uint8)


def depth_2_normal(depth, depth_unvalid):
    B, H, W = depth.shape
    grad_out = torch.zeros((B, H, W, 3)).cuda()

    fx, fy, cx, cy = 262.5, 262.5, 128.0, 128.0

    # Pixel coordinates
    X, Y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing="ij")
    X = (X - cx) / fx
    Y = (Y - cy) / fy
    X = X.repeat(B, 1, 1).cuda()
    Y = Y.repeat(B, 1, 1).cuda()

    X = depth * X
    Y = depth * Y

    XYZ_camera = torch.cat([X[..., None], Y[..., None], depth[..., None]], -1).cuda()

    # compute tangent vectors
    vx = XYZ_camera[:, 1:-1, 2:, :] - XYZ_camera[:, 1:-1, 1:-1, :]
    vy = XYZ_camera[:, 2:, 1:-1, :] - XYZ_camera[:, 1:-1, 1:-1, :]

    # Finally compute the cross product
    normal = torch.cross(vx.reshape(-1, 3), vy.reshape(-1, 3))

    # Avoid division by 0
    normal = torch.where(normal < 1e-5, 1e-5 * torch.ones_like(normal), normal)
    normal_norm = normal.norm(p=2, dim=1, keepdim=True)
    normal_normalized = normal.div(normal_norm)

    # Reshape to image
    normal_out = normal_normalized.reshape(B, H - 2, W - 2, 3)
    grad_out[:, 1:-1, 1:-1, :] = 0.5 - 0.5 * normal_out

    # Zero out +inf
    grad_out[depth_unvalid] = 0.0
    return grad_out


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def lift_2d_to_3d(depth):
    fx, fy = 262.5, 262.5
    cx, cy = 128.0, 128.0

    u, v = torch.where(depth != 0.0)
    y = depth[u, v] * ((u - cy) / fy)
    x = depth[u, v] * ((v - cx) / fx)
    z = depth[u, v]
    pts = torch.stack([x, y, z], axis=-1)
    return pts
