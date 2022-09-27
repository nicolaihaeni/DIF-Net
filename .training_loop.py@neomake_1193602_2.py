# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main training loop for DIF-Net.
"""
import os
import time
import numpy as np
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import utils
import sdf_meshing
from utils import save_checkpoints, to_png, depth_to_png, depth_2_normal
from loss import compute_depth_normal_loss, compute_normal_loss


def train(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    loss_schedules=None,
    is_train=True,
    optim=None,
    model_name=None,
    dataset=None,
    mesh_path=None,
    start_epoch=0,
    gt_points=None,
    symmetry=None,
    **kwargs,
):

    print("Training Info:")
    print("data_path:\t\t", kwargs["root_dir"])
    print("num_instances:\t\t", kwargs["num_instances"])
    print("batch_size:\t\t", kwargs["batch_size"])
    print("epochs:\t\t\t", epochs)
    print("learning rate:\t\t", lr)
    for key in kwargs:
        if "loss" in key:
            print(key + ":\t", kwargs[key])

    if is_train and optim is None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    else:
        embedding = (
            model.latent_codes(torch.zeros(1).long().cuda()).clone().detach()
        )  # initialization for evaluation stage
        embedding.requires_grad = True

        # Also optimize model translations
        translation = torch.Tensor([0.0, 0.0, 0.0]).float().cuda()
        translation.requires_grad = True 

        rotation = torch.Tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float().cuda()
        rotation.requires_grad = True 
        optim = torch.optim.Adam(
            [
                {"params": [embedding], "lr": lr},
                {"params": [translation], "lr": 0.01},
                {"params": [rotation], "lr": 0.01},
            ],
            lr=lr,
        )

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = start_epoch * len(train_dataloader)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        pbar.update(total_steps)
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                if is_train:
                    save_checkpoints(
                        os.path.join(checkpoints_dir, "model_epoch_%04d.pth" % epoch),
                        model,
                        optim,
                        epoch,
                    )
                else:
                    embed_save = embedding.detach().squeeze().cpu().numpy()
                    np.savetxt(
                        os.path.join(
                            checkpoints_dir, "embedding_epoch_%04d.txt" % epoch
                        ),
                        embed_save,
                    )

                np.savetxt(
                    os.path.join(
                        checkpoints_dir, "train_losses_epoch_%04d.txt" % epoch
                    ),
                    np.array(train_losses),
                )

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if is_train:
                    losses = model(model_input, gt, **kwargs)
                else:
                    losses = model.embedding(
                        embedding, translation, rotation, model_input, gt
                    )

                train_loss = 0.0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(
                            loss_name + "_weight",
                            loss_schedules[loss_name](total_steps),
                            total_steps,
                        )
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    if is_train:
                        save_checkpoints(
                            os.path.join(checkpoints_dir, "model_current.pth"),
                            model,
                            optim,
                            epoch,
                        )

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write(
                        "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                        % (epoch, train_loss, time.time() - start_time)
                    )

                total_steps += 1

        if is_train:
            save_checkpoints(
                os.path.join(checkpoints_dir, "model_final.pth"),
                model,
                optim,
                epoch,
            )
        else:
            embed_save = embedding.detach().squeeze().cpu().numpy()
            np.savetxt(
                os.path.join(checkpoints_dir, "embedding_epoch_%04d.txt" % epoch),
                embed_save,
            )
            sdf_meshing.create_mesh(
                model,
                os.path.join(mesh_path, model_name),
                embedding=embedding,
                N=256,
                level=0,
                get_color=False,
            )

            # Load ground truth points and save before/after input point clouds
            import h5py
            import open3d as o3d

            partial_pts = train_dataloader.dataset.partial
            gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_points))
            gt_pcd.paint_uniform_color([1, 0, 0])

            o3d.io.write_point_cloud(
                os.path.join(mesh_path, f"{model_name}_gt.ply"), gt_pcd
            )

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_pts))
            pcd.paint_uniform_color([0, 1, 0])

            o3d.io.write_point_cloud(
                os.path.join(mesh_path, f"{model_name}_before.ply"), pcd
            )

            # Translate the point cloud
            translation = translation.detach().cpu().numpy()
            pcd.translate(translation)
            pcd.paint_uniform_color([0, 0, 1])
            o3d.io.write_point_cloud(
                os.path.join(mesh_path, f"{model_name}_after.ply"), pcd
            )

        np.savetxt(
            os.path.join(checkpoints_dir, "train_losses_final.txt"),
            np.array(train_losses),
        )


def train_depth_model(
    model,
    train_dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    loss_schedules=None,
    normal_model=None,
    is_train=True,
    optim=None,
    model_name=None,
    start_epoch=0,
    dataset=None,
    optimize_normals=False,
    **kwargs,
):

    print("Training Info:")
    print("data_path:\t\t", kwargs["root_dir"])
    print("num_instances:\t\t", kwargs["num_instances"])
    print("batch_size:\t\t", kwargs["batch_size"])
    print("epochs:\t\t\t", epochs)
    print("learning rate:\t\t", lr)

    if is_train and optim is None:
        optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=0.0005)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    if optimize_normals:
        checkpoints_dir = os.path.join(model_dir, "checkpoints", "normals")
    else:
        checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = start_epoch * len(train_dataloader)
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        pbar.update(total_steps)
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                save_checkpoints(
                    os.path.join(checkpoints_dir, "model_epoch_%04d.pth" % epoch),
                    model,
                    optim,
                    epoch,
                )

                np.savetxt(
                    os.path.join(
                        checkpoints_dir, "train_losses_epoch_%04d.txt" % epoch
                    ),
                    np.array(train_losses),
                )

            for step, batch in enumerate(train_dataloader):
                start_time = time.time()

                batch = {key: value.cuda() for key, value in batch.items()}
                gt_depth = batch["depths"]

                if optimize_normals:
                    normal = model(batch["images"], batch["masks"])
                    losses = compute_normal_loss(normal, gt_depth)
                else:
                    normal = normal_model(batch["images"], batch["masks"])
                    depth = model(batch["images"], batch["masks"], normal)
                    losses = compute_depth_normal_loss(depth, normal, gt_depth)

                train_loss = 0.0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(
                            loss_name + "_weight",
                            loss_schedules[loss_name](total_steps),
                            total_steps,
                        )
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    save_checkpoints(
                        os.path.join(checkpoints_dir, "model_current.pth"),
                        model,
                        optim,
                        epoch,
                    )

                optim.zero_grad()
                train_loss.backward()
                optim.step()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write(
                        "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                        % (epoch, train_loss, time.time() - start_time)
                    )

                    # Add sample images to tensorboard
                    writer.add_image(
                        "image",
                        to_png(batch["images"][0]),
                        dataformats="HWC",
                        global_step=epoch,
                    )
                    writer.add_image(
                        "depth",
                        depth_to_png(batch["depths"][0]),
                        dataformats="HW",
                        global_step=epoch,
                    )
                    writer.add_image(
                        "mask",
                        to_png(batch["masks"][0]),
                        dataformats="HW",
                        global_step=epoch,
                    )

                    depth_unvalid = gt_depth == 0.0
                    if not optimize_normals:
                        prediction = depth[0]
                        prediction[batch["masks"][0] == 0.0] = 0.0
                        writer.add_image(
                            "depth_pred",
                            depth_to_png(prediction),
                            dataformats="HW",
                            global_step=epoch,
                        )

                        normal_d = depth_2_normal(
                            depth[0].unsqueeze(0), depth_unvalid[0].unsqueeze(0)
                        ).squeeze(0)
                        normal_d[batch["masks"][0] == 0.0] = 0.0

                        writer.add_image(
                            "normal_depth",
                            to_png(normal_d),
                            dataformats="HWC",
                            global_step=epoch,
                        )

                    normal = normal[0]
                    normal[batch["masks"][0] == 0.0] = 0.0

                    writer.add_image(
                        "normal_intermediate",
                        to_png(normal),
                        dataformats="HWC",
                        global_step=epoch,
                    )

                    normal_gt = depth_2_normal(
                        gt_depth[0].unsqueeze(0), depth_unvalid[0].unsqueeze(0)
                    ).squeeze(0)
                    normal_gt[batch["masks"][0] == 0.0] = 0.0

                    writer.add_image(
                        "normal",
                        to_png(normal_gt),
                        dataformats="HWC",
                        global_step=epoch,
                    )

                total_steps += 1

        save_checkpoints(
            os.path.join(checkpoints_dir, "model_final.pth"),
            model,
            optim,
            epochs,
        )

        np.savetxt(
            os.path.join(checkpoints_dir, "train_losses_final.txt"),
            np.array(train_losses),
        )
