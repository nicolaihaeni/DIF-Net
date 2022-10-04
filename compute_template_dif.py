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
from datasets.shapenet import ShapenetEvalDataset
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

file_names = utils.get_filenames(
    meta_params["root_dir"], meta_params["split_file"], mode="test"
)

cvs_columns = ["name", "chamfer", "f1"]
dict_data = []

recon_name = (
    "/home/isleri/haeni001/code/DIF-Net/logs/plane/recon/train/template_0.0.ply"
)
mesh = trimesh.load(recon_name)
recon_points = np.array(trimesh.sample.sample_surface(mesh, 100000)[0])
for ii, filename in enumerate(file_names):
    print(filename)
    basename = os.path.basename(filename).split(".")[0]

    gt_path = os.path.join(meta_params["gt_dir"], basename, f"{basename}.h5")
    with h5py.File(gt_path, "r") as hf:
        gt_points = hf["surface_pts"][:, :3]

    cd, f1 = compute_recon_error_pts(recon_points, gt_points)
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

with open("./output_metrics.csv", "w") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=cvs_columns)
    writer.writeheader()
    for data in dict_data:
        writer.writerow(data)