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

import utils
from datasets.shapenet import ShapenetMultiDataset
from dif_net import DeformedImplicitField
import sdf_meshing


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

p = configargparse.ArgumentParser()
p.add_argument("--logging_root", type=str, default="./logs", help="root for logging")
p.add_argument("--config", required=True, help="generation configuration")
p.add_argument(
    "--level", type=float, default=0, help="level of iso-surface for marching cube"
)
p.add_argument(
    "--num_instances", type=int, default=10, help="number of instances to generate"
)

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config), "r") as stream:
    meta_params = yaml.safe_load(stream)

# define dataloader
test_dataset = ShapenetMultiDataset(
    utils.get_filenames(meta_params["root_dir"], meta_params["split_file"]),
    on_surface_points=1000,
    max_points=1000,
)

dataloader = DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=1,
    pin_memory=True,
    num_workers=0,
    drop_last=False,
    collate_fn=test_dataset.collate_fn,
)

print("Total subjects: ", len(test_dataset))

# define DIF-Net
model = DeformedImplicitField(**meta_params).cuda()
model.load_state_dict(torch.load(meta_params["checkpoint_path"])["model_state_dict"])
model.eval()

# create save path
root_path = os.path.join(opt.logging_root, meta_params["experiment_name"])
utils.cond_mkdir(root_path)

# create the output mesh directory
mesh_path = os.path.join(root_path, "recon", "train")
utils.cond_mkdir(mesh_path)

# generate meshes with color-coded coordinates
for step, (model_input, gt) in enumerate(dataloader):
    print("generate_instance:", step)
    model_input = {key: value.cuda() for key, value in model_input.items()}
    gt = {key: value.cuda() for key, value in gt.items()}

    # # Save the input point cloud
    # sdf_meshing.save_poincloud_ply(
    # model_input["farthest_points"].squeeze(0).detach().cpu().numpy(),
    # os.path.join(mesh_path, f"model_{step}_input.ply"),
    # )

    embedding = model.get_latent_code(torch.tensor([step]).cuda())

    # Save the ouput mesh
    sdf_meshing.create_mesh(
        model,
        os.path.join(mesh_path, f"model_{step}_prediction"),
        embedding=embedding,
        N=256,
        level=opt.level,
    )

    if step == opt.num_instances:
        break

# Save the template with varying thresholds
threshs = [0.0, 0.01, 0.02]
for thresh in threshs:
    # Save the ouput mesh
    sdf_meshing.create_mesh(
        model,
        os.path.join(mesh_path, f"template_{thresh}"),
        N=256,
        level=thresh,
        template=True,
    )
