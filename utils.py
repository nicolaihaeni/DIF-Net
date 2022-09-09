# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import numpy as np
import torch
import numpy as np


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


def load_checkpoints(args, model, optimizer=None):
    """Load model, optimzer and global iteration from file"""
    start = 0

    # Load checkpoints
    experiment_path = os.path.join(
        args["logging_root"], args["experiment_name"], "checkpoints"
    )

    if not os.path.exists(experiment_path):
        return start, model, optimizer

    ckpts = [
        os.path.join(experiment_path, f)
        for f in sorted(os.listdir(experiment_path))
        if f.endswith(".tar")
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


def get_filenames(root_dir, split_file, mode="train"):
    with open(split_file, "r") as in_file:
        data = json.load(in_file)[mode]

    instances = []
    for cat in data:
        for filename in data[cat]:
            instances.append(os.path.join(root_dir, filename, f"{filename}.h5"))
    return instances


def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)


def sample_spherical(n, radius=3.0):
    xyz = np.random.normal(size=(n, 3))
    xyz = normalize(xyz) * radius
    return xyz
