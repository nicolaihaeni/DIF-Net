# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

"""Evaluation script for DIF-Net.
"""
import io
import os
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import yaml
import random
import trimesh
import numpy as np
from datasets.ddap import DDAPDataset
import utils, training_loop, loss, modules, meta_modules

import torch
from torch import nn
from torch.utils.data import DataLoader
import configargparse
from dif_net import DeformedImplicitField
from calculate_chamfer_distance import compute_recon_error_pts


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


p = configargparse.ArgumentParser()
p.add_argument("--config", required=True, help="Evaluation configuration")
p.add_argument(
    "--use_gt_poses", action="store_true", help="Comparison using ground truth poses"
)
p.add_argument("--symmetry", action="store_true", help="Enforce symmetry")

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config), "r") as stream:
    meta_params = yaml.safe_load(stream)

meta_params["expand"] = 0

# define DIF-Net
model = DeformedImplicitField(**meta_params)
model.load_state_dict(torch.load(meta_params["checkpoint_path"]))

# The network should be fixed for evaluation.
for param in model.template_field.parameters():
    param.requires_grad = False
for param in model.hyper_net.parameters():
    param.requires_grad = False

model.cuda()

# create save path
experiment_path = os.path.join(
    meta_params["logging_root"], meta_params["experiment_name"]
)
utils.cond_mkdir(experiment_path)

# create the output mesh directory
if opt.symmetry:
    mesh_path = os.path.join(experiment_path, "recon", "symmetry")
else:
    mesh_path = os.path.join(experiment_path, "recon", "ours")


utils.cond_mkdir(mesh_path)
meta_params["mesh_path"] = mesh_path
file_names = os.listdir(meta_params["root_dir"])

# Get camera poses for simulated dropout
cvs_columns = ["name", "chamfer", "f1"]
dict_data = []
for ii, filename in enumerate(file_names):
    print(filename)
    basename = os.path.basename(filename).split(".")[0]

    data = np.load(
        os.path.join(meta_params["root_dir"], filename), allow_pickle=True
    ).item()
    scale = data["scale"]
    partial_points = data["partial_points"]
    cam_pose = data["est_w_T_cam"]

    # if already embedded, pass
    recon_name = os.path.join(mesh_path, f"{basename}.ply")
    if os.path.isfile(recon_name):
        print(f"File {basename}.ply exists. Skipping...")
    else:
        # load ground truth data
        sdf_dataset = DDAPDataset(
            partial_points=partial_points,
            cam_pose=cam_pose,
            on_surface_points=4000,
            scale=scale,
        )

        dataloader = DataLoader(
            sdf_dataset,
            shuffle=True,
            batch_size=1,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )

        # shape embedding
        training_loop.train(
            model=model,
            train_dataloader=dataloader,
            model_dir=experiment_path,
            model_name=basename,
            is_train=False,
            dataset=sdf_dataset,
            gt_pose=opt.use_gt_poses,
            **meta_params,
        )
