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


def train(
    model,
    train_dataloader,
    val_dataloader,
    epochs,
    lr,
    steps_til_summary,
    steps_til_validation,
    epochs_til_checkpoint,
    model_dir,
    mesh_dir,
    loss_schedules=None,
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
            print(key + ":\t\t", kwargs[key])

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
                torch.save(
                    model.module.state_dict(),
                    os.path.join(checkpoints_dir, "model_epoch_%04d.pth" % epoch),
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

                losses = model(model_input, gt, **kwargs)

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

                optim.zero_grad()
                train_loss.backward()
                optim.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write(
                        "Epoch %d, Total loss %0.6f, iteration time %0.6f"
                        % (epoch, train_loss, time.time() - start_time)
                    )

                if not total_steps % steps_til_summary:
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(checkpoints_dir, "model_current.pth"),
                    )

                if not total_steps % steps_til_validation:
                    # First we save the deformed mesh
                    model.eval()
                    val_losses = []
                    for step, (model_input, gt) in enumerate(val_dataloader):
                        start_time = time.time()

                        model_input = {
                            key: value.cuda() for key, value in model_input.items()
                        }
                        gt = {key: value.cuda() for key, value in gt.items()}

                        losses = model(model_input, gt, **kwargs)

                        val_loss = 0.0
                        for loss_name, loss in losses.items():
                            single_loss = loss.detach().cpu().mean()

                            if (
                                loss_schedules is not None
                                and loss_name in loss_schedules
                            ):
                                single_loss *= loss_schedules[loss_name](total_steps)
                            val_loss += single_loss
                        val_losses.append(val_loss)
                    val_loss = sum(val_losses) / len(val_losses)
                    writer.add_scalar("val_loss", val_loss, total_steps)
                    tqdm.write(f"Epoch {epoch} Val loss {val_loss}")

                    sdf_meshing.create_mesh(
                        model.module,
                        os.path.join(mesh_dir, f"deformed_mesh_{total_steps}"),
                        model_input,
                    )

                    # Then we also save the template mesh
                    sdf_meshing.create_mesh(
                        model.module,
                        os.path.join(mesh_dir, f"template_mesh_{total_steps}"),
                        model_input,
                        template=True,
                    )
                    model.train()

                total_steps += 1

        torch.save(
            model.module.cpu().state_dict(),
            os.path.join(checkpoints_dir, "model_final.pth"),
        )

    np.savetxt(
        os.path.join(checkpoints_dir, "train_losses_final.txt"),
        np.array(train_losses),
    )
