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
from unet import DepthNet
import imageio


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
            imgs = hf["rgb"][0] / 255.0
            masks = hf["mask"][0]
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
    model = DepthNet([1], ["depth"], 4).cuda()

    # Check if model should be resumed
    start, model, optim = utils.load_checkpoints(meta_params, model)

    # Run the images through the depth prediction network
    inputs = torch.cat(
        [torch.tensor(imgs)[None], torch.tensor(masks)[None, ..., None]], -1
    ).permute(0, 3, 1, 2)
    depth = model(inputs.cuda())["depth"]

    import open3d as o3d

    depth = depth[0].detach().cpu().numpy()
    depth[masks == 0] = np.inf

    u, v = np.where(depth != np.inf)
    y = depth[u, v] * ((u - 128.0) / 262.5)
    x = depth[u, v] * ((v - 128.0) / 262.5)
    z = depth[u, v]
    pts = np.stack([x, y, z], axis=-1)

    pcd_part = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd_part.paint_uniform_color(np.array([0, 1, 0]))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([axis, pcd_part])
