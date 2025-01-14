# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Training script for DIF-Net.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import io
import utils, training_loop
from datasets.shapenet import ShapenetMultiDataset

from torch.utils.data import DataLoader
import configargparse
from torch import nn
from dif_net import DeformedImplicitField


if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    p.add_argument("--config", type=str, default="", help="training configuration.")
    p.add_argument("--root_dir", type=str, default="", help="training data path.")
    p.add_argument("--split_file", type=str, default="", help="training subject names.")

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

    p.add_argument(
        "--epochs_til_checkpoint",
        type=int,
        default=5,
        help="Time interval in seconds until checkpoint is saved.",
    )
    p.add_argument(
        "--steps_til_summary",
        type=int,
        default=100,
        help="Time interval in iterations until tensorboard summary is saved.",
    )
    p.add_argument(
        "--steps_til_validation",
        type=int,
        default=1000,
        help="Time interval in iterations until output meshes are saved.",
    )

    p.add_argument(
        "--model_type",
        type=str,
        default="sine",
        help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)',
    )

    p.add_argument("--latent_dim", type=int, default=128, help="latent code dimension.")
    p.add_argument(
        "--hidden_num",
        type=int,
        default=128,
        help="hidden layer dimension of deform-net.",
    )
    p.add_argument(
        "--loss_grad_deform",
        type=float,
        default=5,
        help="loss weight for deformation smoothness prior.",
    )
    p.add_argument(
        "--loss_grad_temp",
        type=float,
        default=1e2,
        help="loss weight for normal consistency prior.",
    )
    p.add_argument(
        "--loss_correct",
        type=float,
        default=1e2,
        help="loss weight for minimal correction prior.",
    )

    p.add_argument(
        "--expand", type=float, default=-1, help="expansion of shape surface."
    )
    p.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="number of surface points for each epoch.",
    )
    p.add_argument(
        "--on_surface_points",
        type=int,
        default=4000,
        help="number of surface points for each iteration.",
    )

    # load configs if exist
    opt = p.parse_args()
    if opt.config == "":
        meta_params = vars(opt)
    else:
        with open(opt.config, "r") as stream:
            meta_params = yaml.safe_load(stream)

    # define dataloader
    train_dataset = ShapenetMultiDataset(
        utils.get_filenames(meta_params["root_dir"], meta_params["split_file"]),
        on_surface_points=opt.on_surface_points,
        max_points=meta_params["max_points"],
        expand=meta_params["expand"],
        train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=meta_params["batch_size"],
        pin_memory=True,
        num_workers=24,
        drop_last=True,
        prefetch_factor=8,
        collate_fn=train_dataset.collate_fn,
    )

    print("Total subjects: ", train_dataset.num_instances)
    meta_params["num_instances"] = train_dataset.num_instances

    # create save path
    root_path = os.path.join(
        meta_params["logging_root"], meta_params["experiment_name"]
    )
    utils.cond_mkdir(root_path)

    with io.open(os.path.join(root_path, "model.yml"), "w", encoding="utf8") as outfile:
        yaml.dump(meta_params, outfile, default_flow_style=False, allow_unicode=True)

    # define DIF-Net
    model = DeformedImplicitField(**meta_params)
    model = nn.DataParallel(model).cuda()

    # Check if model should be resumed
    start, model, optim = utils.load_checkpoints(meta_params, model)

    # main decoder training loop
    training_loop.train(
        model=model,
        optim=optim,
        start_epoch=start,
        train_dataloader=train_loader,
        model_dir=root_path,
        **meta_params
    )
