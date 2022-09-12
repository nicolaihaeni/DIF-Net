# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main training loop for DIF-Net.
"""
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import sdf_meshing
from utils import save_checkpoints, to_png, depth_to_png


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
    mesh_path=None,
    dataset=None,
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
        optim = torch.optim.Adam(lr=lr, params=[embedding])

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
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
                    losses = model.embedding(embedding, model_input, gt)

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

            # Save the ground truth and partial point cloud to files
            partial_pcd, gt_pcd = dataset.get_point_clouds(0)
            sdf_meshing.save_poincloud_ply(
                gt_pcd,
                model,
                embedding,
                os.path.join(mesh_path, f"{model_name}_gt.ply"),
            )
            sdf_meshing.save_poincloud_ply(
                partial_pcd,
                model,
                embedding,
                os.path.join(mesh_path, f"{model_name}_partial.ply"),
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
    is_train=True,
    optim=None,
    model_name=None,
    mesh_path=None,
    dataset=None,
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

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, "summaries")
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
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

                model_input = {key: value.cuda() for key, value in model_input.items()}
                inputs = torch.cat(
                    [model_input["images"], model_input["masks"][..., None]], -1
                ).permute(0, 3, 1, 2)
                gt = model_input["depths"]
                model_outputs = model(inputs, gt)
                train_loss = model_outputs["depth_loss"].mean()

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                # Add sample images to tensorboard
                writer.add_image(
                    "image",
                    model_input["images"][0],
                    dataformat="HWC",
                    global_step=total_steps,
                )
                writer.add_image(
                    "depth",
                    model_input["depths"][0],
                    dataformat="HW",
                    global_step=total_steps,
                )
                writer.add_image(
                    "mask",
                    model_input["masks"][0],
                    dataformat="HW",
                    global_step=total_steps,
                )
                writer.add_image(
                    "prediction",
                    model_outputs["depth"][0],
                    dataformat="HW",
                    global_step=total_steps,
                )

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

                total_steps += 1

        save_checkpoints(
            os.path.join(checkpoints_dir, "model_final.pth"),
            model,
            optim,
            epoch,
        )

        np.savetxt(
            os.path.join(checkpoints_dir, "train_losses_final.txt"),
            np.array(train_losses),
        )
