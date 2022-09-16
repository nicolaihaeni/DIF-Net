#  Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Training losses for DIF-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import depth_2_normal


def deform_implicit_loss(
    model_output, gt, loss_grad_deform=5, loss_grad_temp=1e2, loss_correct=1e2
):

    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    embeddings = model_output["latent_vec"]

    gradient_sdf = model_output["grad_sdf"]
    gradient_deform = model_output["grad_deform"]
    gradient_temp = model_output["grad_temp"]
    sdf_correct = model_output["sdf_correct"]

    # sdf regression loss from Sitzmannn et al. 2020
    sdf_constraint = torch.where(
        gt_sdf != -1,
        torch.clamp(pred_sdf, -0.5, 0.5) - torch.clamp(gt_sdf, -0.5, 0.5),
        torch.zeros_like(pred_sdf),
    )
    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf))
    )
    normal_constraint = torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient_sdf[..., :1]),
    )
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)

    # deformation smoothness prior
    grad_deform_constraint = gradient_deform.norm(dim=-1)

    # normal consistency prior
    grad_temp_constraint = torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(gradient_temp, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient_temp[..., :1]),
    )

    # minimal correction prior
    sdf_correct_constraint = torch.abs(sdf_correct)

    # latent code prior
    embeddings_constraint = torch.mean(embeddings**2)

    # -----------------
    return {
        "sdf": torch.abs(sdf_constraint).mean() * 3e3,
        "inter": inter_constraint.mean() * 5e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
        "embeddings_constraint": embeddings_constraint.mean() * 1e6,
        "grad_temp_constraint": grad_temp_constraint.mean() * loss_grad_temp,
        "grad_deform_constraint": grad_deform_constraint.mean() * loss_grad_deform,
        "sdf_correct_constraint": sdf_correct_constraint.mean() * loss_correct,
    }


def embedding_loss(model_output, gt):
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    pred_sdf = model_output["model_out"]

    embeddings = model_output["latent_vec"]
    gradient_sdf = model_output["grad_sdf"]

    # sdf regression loss from Sitzmannn et al. 2020
    sdf_constraint = torch.where(
        gt_sdf != -1,
        torch.clamp(pred_sdf, -0.5, 0.5) - torch.clamp(gt_sdf, -0.5, 0.5),
        torch.zeros_like(pred_sdf),
    )
    inter_constraint = torch.where(
        gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf))
    )
    normal_constraint = torch.where(
        gt_sdf == 0,
        1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
        torch.zeros_like(gradient_sdf[..., :1]),
    )
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)

    embeddings_constraint = torch.mean(embeddings**2)

    # -----------------
    return {
        "sdf": torch.abs(sdf_constraint).mean() * 3e3,
        "inter": inter_constraint.mean() * 5e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
        "embeddings_constraint": embeddings_constraint.mean() * 1e6,
    }


def compute_normal_loss(normal, depth_gt):
    depth_unvalid = depth_gt == 0.0  # excludes background
    is_fg = depth_gt != 0.0  # excludes background
    normal_gt = depth_2_normal(depth_gt, depth_unvalid)

    loss_norm = cosine_loss(normal[is_fg], normal_gt[is_fg])
    return {"normal_loss": loss_norm}


def compute_depth_normal_loss(depth_pred, normal_pred, depth_gt):
    depth_unvalid = depth_gt == 0.0  # excludes background
    is_fg = depth_gt != 0.0  # excludes background

    # Compute normal maps from depth
    normal_d = depth_2_normal(depth_pred, depth_unvalid)
    normal_gt = depth_2_normal(depth_gt, depth_unvalid)

    loss_depth = depth_loss(depth_pred[is_fg], depth_gt[is_fg])
    loss_norm = cosine_loss(normal_d[is_fg], normal_gt[is_fg])
    loss_refine = refined_loss(depth_pred[is_fg], depth_gt[is_fg])

    return {
        "depth_loss": 2 * loss_depth,
        "normal_loss": loss_norm,
        "refined_loss": loss_refine,
    }


def depth_loss(depth_pred, depth_gt):
    return F.mse_loss(depth_pred, depth_gt)


def cosine_loss(normal_pred, normal_gt):
    normals_fg = normal_pred[..., None].permute(0, 2, 1)  # n^T, Nx1x3
    normals_fg_gt = normal_gt[..., None]  # n, Nx3x1

    nominator = torch.bmm(normals_fg, normals_fg_gt).squeeze(-1)
    denominator = normals_fg.norm(p=2, dim=-1) * normals_fg_gt.norm(p=2, dim=1)

    cos_angle = nominator / (denominator + 1e-5)
    normal_loss = torch.acos(cos_angle).mean()
    return normal_loss


def refined_loss(depth, depth_gt):
    gt_max = torch.max(depth_gt)

    pred_max = torch.max(depth)
    pred_min = torch.min(depth)

    # Normalize depth
    depth = gt_max * (depth - pred_min) / (pred_max - pred_min)

    d = depth - depth_gt
    a1 = torch.sum(d**2)
    a2 = torch.sum(d) ** 2
    length = d.shape[0]
    return torch.mean(torch.div(a1, length) - (0.5 * torch.div(a2, length**2)))
