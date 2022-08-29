# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoints(path, model, optimizer, global_step):
    """Save model, optimzer and global iteration to file"""
    if isinstance(model, torch.nn.DataParallel):
        torch.save(
            {
                "epoch": global_step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )
    else:
        torch.save(
            {
                "epoch": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path,
        )


def load_checkpoints(args, model, optimizer=None, name="decoder"):
    """Load model, optimzer and global iteration from file"""
    start = 0

    # Load checkpoints
    experiment_path = os.path.join(
        args["logging_root"], args["experiment_name"], "checkpoints"
    )

    if not os.path.exists(experiment_path):
        return start, model, optimizer

    ckpts = [
        os.path.join(experiment_path, f)
        for f in sorted(os.listdir(experiment_path))
        if f.endswith(".tar") and name in f
    ]

    print(f"Found checkpoints {ckpts}")
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print(f"Reloading checkpoints from {ckpt_path}")
        ckpt = torch.load(ckpt_path)

        start = ckpt["epoch"]
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Load model
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
    return start, model, optimizer
