# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generation script for DIF-Net.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import configargparse
import re
import numpy as np

import torch
from torch.utils.data import DataLoader

import dataset, modules, utils
from dif_net import DeformedImplicitField
import sdf_meshing


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

p = configargparse.ArgumentParser()
p.add_argument("--logging_root", type=str, default="./logs", help="root for logging")
p.add_argument("--config", required=True, help="generation configuration")
p.add_argument(
    "--level", type=float, default=0, help="level of iso-surface for marching cube"
)

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config), "r") as stream:
    meta_params = yaml.safe_load(stream)

# define dataloader
sdf_dataset = dataset.PointCloudSingleDataset(
    root_dir=meta_params["root_dir"],
    split_file=meta_params["split_file"],
    on_surface_points=1000,
    # train=True,
)

dataloader = DataLoader(
    sdf_dataset,
    shuffle=False,
    batch_size=1,
    pin_memory=True,
    num_workers=4,
    drop_last=False,
)

print("Total subjects: ", len(sdf_dataset))
meta_params["num_instances"] = len(sdf_dataset)

# define DIF-Net
model = DeformedImplicitField(**meta_params)
model.load_state_dict(torch.load(meta_params["checkpoint_path"]))
model.cuda()
model.eval()

# create save path
root_path = os.path.join(opt.logging_root, meta_params["experiment_name"])
utils.cond_mkdir(root_path)

# create the output mesh directory
mesh_path = os.path.join(root_path, "recon", "val")
utils.cond_mkdir(mesh_path)

# generate meshes with color-coded coordinates
for step, (model_input, gt) in enumerate(dataloader):
    print("generate_instance:", step)
    model_input = {key: value.cuda() for key, value in model_input.items()}
    gt = {key: value.cuda() for key, value in gt.items()}

    # Save the input point cloud
    sdf_meshing.save_poincloud_ply(
        model_input["farthest_points"],
        os.path.join(mesh_path, f"{step}_input.ply"),
    )

    # Save the ouput mesh
    sdf_meshing.create_mesh(
        model,
        os.path.join(mesh_path, f"{step}_prediction"),
        model_input,
        N=256,
        level=opt.level,
    )
    if step >= 10:
        break

# Save the template with varying thresholds
threshs = [0.0, 0.1, 0.2]
for thresh in threshs:
    # Save the ouput mesh
    sdf_meshing.create_mesh(
        model,
        os.path.join(mesh_path, f"template_{thresh}"),
        model_input,
        N=256,
        level=thresh,
        template=True,
    )
