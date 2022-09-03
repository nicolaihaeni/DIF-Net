# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluation script for DIF-Net.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import io
import numpy as np
import dataset, utils, training_loop, loss, modules, meta_modules

import torch
from torch.utils.data import DataLoader
import configargparse
from torch import nn
from pointnet import Encoder
from dif_net import DeformedImplicitField
from calculate_chamfer_distance import compute_recon_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

p = configargparse.ArgumentParser()
p.add_argument("--config", required=True, help="Evaluation configuration")

# load configs
opt = p.parse_args()
with open(os.path.join(opt.config), "r") as stream:
    meta_params = yaml.safe_load(stream)

meta_params["expand"] = 0

# define DIF-Net
model = DeformedImplicitField(**meta_params).cuda()
ckpt = torch.load(meta_params["checkpoint_path"])
model.load_state_dict(ckpt["model_state_dict"])

# Define the encoder
encoder = Encoder(latent_dim=meta_params["latent_dim"]).cuda()
ckpt = torch.load(meta_params["encoder_checkpoint_path"])
encoder.load_state_dict(ckpt["model_state_dict"])

encoder.eval()
model.eval()

# The network should be fixed for evaluation.
for param in model.template_field.parameters():
    param.requires_grad = False
for param in model.hyper_net.parameters():
    param.requires_grad = False

# create save path
root_path = os.path.join(
    meta_params["logging_root"], meta_params["experiment_name"], "eval"
)
utils.cond_mkdir(root_path)

# load names for evaluation subjects
all_names = (utils.get_filenames(meta_params["root_dir"], meta_params["split_file"]),)

# optimize latent code for each test subject
for file in all_names:
    print(file)
    save_path = os.path.join(root_path, file)

    # if already embedded, pass
    if os.path.isfile(os.path.join(save_path, "test.ply")):
        continue

    # load ground truth data
    sdf_dataset = dataset.PointCloudMulti(
        filename=[file], max_num_instances=-1, **meta_params
    )
    dataloader = DataLoader(
        sdf_dataset,
        shuffle=True,
        collate_fn=sdf_dataset.collate_fn,
        batch_size=1,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
    )

    # shape embedding
    training_loop.fit_latent_code(
        model=model,
        train_dataloader=dataloader,
        model_dir=save_path,
        is_train=False,
        **meta_params
    )

# calculate chamfer distance for each subject
# chamfer_dist = []
# for file in all_names:
# recon_name = os.path.join(root_path, file, "checkpoints", "test.ply")
# gt_name = os.path.join(meta_params["point_cloud_path"], file + ".mat")
# cd = compute_recon_error(recon_name, gt_name)
# print(file, "\tcd:%f" % cd)
# chamfer_dist.append(cd)

# print("Average Chamfer Distance:", np.mean(chamfer_dist))
