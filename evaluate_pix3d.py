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
from datasets.pix3d import Pix3dDataset
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
if opt.use_gt_poses:
    mesh_path = os.path.join(experiment_path, "recon", "test", "gt_pose")
elif "equi" in meta_params["camera_file"]:
    mesh_path = os.path.join(experiment_path, "recon", "test", "equi_pose")
else:
    mesh_path = os.path.join(experiment_path, "recon", "test", "ours_pose")

utils.cond_mkdir(mesh_path)
meta_params["mesh_path"] = mesh_path
file_names = utils.get_pix3d_filenames(os.path.join(meta_params["root_dir"]))

# Get camera poses for simulated dropout
cam_poses = np.load(
    os.path.join(meta_params["root_dir"], meta_params["camera_file"]),
    allow_pickle=True,
).item()

cvs_columns = ["name", "chamfer", "f1"]
dict_data = []
for ii, filename in enumerate(file_names):
    print(filename)
    basename = os.path.basename(filename).split(".")[0]

    names = cam_poses["names"]
    index = names.index(basename)
    gt_cam_pose = np.linalg.inv(cam_poses["gt_cam_T_w"][index])
    est_cam_pose = cam_poses["est_w_T_cam"][index]

    # Get the right camera pose
    with h5py.File(os.path.join(meta_params["gt_dir"], filename), "r") as hf:
        scale = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        gt_points = (scale @ (hf["gt_pts"][:, :3]).transpose()).transpose()

    if opt.use_gt_poses:
        cam_pose = gt_cam_pose
    else:
        cam_pose = est_cam_pose

    # if already embedded, pass
    recon_name = os.path.join(mesh_path, f"{basename}.ply")
    if os.path.isfile(recon_name):
        print(f"File {basename}.ply exists. Skipping...")
    else:
        # load ground truth data
        sdf_dataset = Pix3dDataset(
            input_file_name=filename,
            cam_pose=cam_pose,
            on_surface_points=4000,
            symmetry=opt.symmetry,
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
            gt_points=gt_points,
            gt_pose=opt.use_gt_poses,
            **meta_params,
        )

for ii, filename in enumerate(file_names):
    print(filename)
    basename = os.path.basename(filename).split(".")[0]

    with h5py.File(os.path.join(meta_params["gt_dir"], filename), "r") as hf:
        scale = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        gt_points = (scale @ (hf["gt_pts"][:, :3]).transpose()).transpose()

    recon_path = os.path.join(mesh_path, f"{basename}.ply")
    recon_mesh = trimesh.load(recon_path)
    if isinstance(recon_mesh, trimesh.Scene):
        recon_mesh = recon_mesh.dump().sum()

    recon_pts = np.array(trimesh.sample.sample_surface(recon_mesh, 100000)[0])
    cd, f1 = compute_recon_error_pts(recon_pts, gt_points)
    dict_data.append({"name": filename, "chamfer": cd, "f1": f1})

chamfer = [f["chamfer"] for f in dict_data]
f1 = [f["f1"] for f in dict_data]
print("Average Chamfer Distance:", 1e4 * np.mean(np.array(chamfer)))
print("Average F1 Score @1:", np.mean(np.array(f1)))
dict_data.append(
    {
        "name": "Total",
        "chamfer": 1e4 * np.mean(np.array(chamfer)),
        "f1": np.mean(np.array(f1)),
    }
)

with open(os.path.join(mesh_path, "output_metrics.csv"), "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=cvs_columns)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)
