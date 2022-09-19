# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

"""Evaluation script for DIF-Net.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import io
import random
import numpy as np
import dataset, utils, training_loop, loss, modules, meta_modules

import torch
from torch import nn
from torch.utils.data import DataLoader
import configargparse
import sdf_meshing
from dif_net import DeformedImplicitField
from calculate_chamfer_distance import compute_recon_error


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
p.add_argument("--sym", action="store_true", help="Enforce symmetry")

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
if use_gt_poses:
    mesh_path = os.path.join(experiment_path, "recon", "test", "gt_pose")
else:
    mesh_path = os.path.join(experiment_path, "recon", "test", "est_pose")

utils.cond_mkdir(mesh_path)
meta_params["mesh_path"] = mesh_path

file_names = utils.get_filenames(
    meta_params["root_dir"], meta_params["split_file"], mode="test"
)

# Get camera poses for simulated dropout
cam_poses = np.load(
    os.path.join(meta_params["root_dir"], meta_params["camera_file"]),
    allow_pickle=True,
).item()

chamfer_dist, f1_score = [], []
for ii, filename in enumerate(file_names):
    print(filename)
    basename = os.path.basename(filename).split(".")[0]

    # Get the right camera pose
    index = cam_poses["names"].index(basename)
    if opt.use_gt_poses:
        cam_pose = cam_poses["gt_w_T_cam"][index]
    else:
        cam_pose = cam_poses["est_w_T_cam"][index]

    # if already embedded, pass
    recon_name = os.path.join(mesh_path, f"{basename}.ply")
    if os.path.isfile(recon_name):
        print(f"File {basename}.ply exists. Skipping...")
    else:
        # load ground truth data
        sdf_dataset = dataset.PointCloudEvalDataset(
            input_file_name=filename,
            cam_pose=gt_cam_pose,
            on_surface_points=4000,
            symmetry=False,
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
            **meta_params,
        )

        cd, f1 = compute_recon_error(
            recon_name,
            os.path.join(meta_params["gt_dir"], basename, f"{basename}.h5"),
        )
        chamfer_dist.append(cd)
        f1_score.append(f1)

print("Average Chamfer Distance:", 1e4 * np.mean(chamfer_dist))
print("Average F1 Score @1:", np.mean(f1_score))
