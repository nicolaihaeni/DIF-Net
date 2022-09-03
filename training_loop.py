# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main training loop for DIF-Net.
"""
import os
import time
import numpy as np
import torch
import utils
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import sdf_meshing


def train(
    model,
    optim,
    start_epoch,
    dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
    loss_schedules=None,
    encoder=None,
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

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if encoder is None:
        summaries_dir = os.path.join(model_dir, "summaries", "decoder")
    else:
        summaries_dir = os.path.join(model_dir, "summaries", "encoder")

    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = start_epoch * len(dataloader)
    print("Start training the decoder...")
    with tqdm(total=len(dataloader) * epochs) as pbar:
        train_losses = []
        pbar.update(total_steps)
        for epoch in range(start_epoch, epochs):
            for step, (model_input, gt) in enumerate(dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if encoder is None:
                    losses = model(model_input, gt, **kwargs)
                else:
                    embedding = encoder(model_input["farthest_points"])
                    losses = model.module.fit_encoder(embedding, model_input, gt)

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

                total_steps += 1

            if not epoch % epochs_til_checkpoint and epoch:
                prefix = "decoder" if encoder is None else "encoder"
                utils.save_checkpoints(
                    os.path.join(checkpoints_dir, f"{prefix}_epoch_{epoch}.tar"),
                    model,
                    optim,
                    epoch,
                )

                np.savetxt(
                    os.path.join(
                        checkpoints_dir, "{prefix}_train_losses_epoch_%04d.txt" % epoch
                    ),
                    np.array(train_losses),
                )

        utils.save_checkpoints(
            os.path.join(checkpoints_dir, "{prefix}_final.tar"),
            model,
            optim,
            epochs,
        )

    np.savetxt(
        os.path.join(checkpoints_dir, "{prefix}_train_losses_final.txt"),
        np.array(train_losses),
    )


def fit_latent_code(
    model,
    encoder,
    embedding,
    optim,
    dataloader,
    epochs,
    lr,
    steps_til_summary,
    epochs_til_checkpoint,
    model_dir,
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

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Do an initial embedding of the points using the encoder
    data = next(dataloader)
    embedding = encoder(data["farthest_points"].cuda())
    embedding = torch.nn.Embedding().from_pretrained(
        embedding.detach().cpu(), freeze=False
    )

    summaries_dir = os.path.join(model_dir, "summaries", "latent")
    utils.cond_mkdir(summaries_dir)

    writer = SummaryWriter(summaries_dir)
    total_steps = len(dataloader)
    print("Start fitting the latent code...")
    with tqdm(total=len(dataloader) * epochs) as pbar:
        train_losses = []
        pbar.update(total_steps)
        for epoch in range(epochs):
            for step, (model_input, gt) in enumerate(dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    np.save(
        os.path.join(model_dir, f"epoch_{epoch}_latent.npy"),
        embedding.detach().cpu().numpy(),
    )

    sdf_meshing.create_mesh(
        model,
        os.path.join(model_dir, "test"),
        embedding=embedding,
        N=256,
        level=0.0,
    )
