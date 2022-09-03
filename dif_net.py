# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define DIF-Net
"""

import torch
from torch import nn
import torch.nn.functional as F
import modules
from meta_modules import HyperNetwork
from loss import *


class DeformedImplicitField(nn.Module):
    def __init__(
        self,
        num_instances,
        latent_dim=128,
        model_type="sine",
        hyper_hidden_layers=1,
        hyper_hidden_features=256,
        hidden_num=128,
        train=True,
        **kwargs,
    ):
        super().__init__()
        # self.train = train
        # self.latent_dim = latent_dim
        # self.encoder = Encoder(latent_dim)

        # latent code embedding for training subjects
        self.latent_dim = latent_dim
        self.latent_codes = nn.Embedding(num_instances, self.latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

        ## DIF-Net
        # template field
        self.template_field = modules.SingleBVPNet(
            type=model_type,
            mode="mlp",
            hidden_features=hidden_num,
            num_hidden_layers=3,
            in_features=3,
            out_features=1,
        )

        # Deform-Net
        self.deform_net = modules.SingleBVPNet(
            type=model_type,
            mode="mlp",
            hidden_features=hidden_num,
            num_hidden_layers=3,
            in_features=3,
            out_features=4,
        )

        # Hyper-Net
        self.hyper_net = HyperNetwork(
            hyper_in_features=self.latent_dim,
            hyper_hidden_layers=hyper_hidden_layers,
            hyper_hidden_features=hyper_hidden_features,
            hypo_module=self.deform_net,
        )
        print(self)

    def get_hypo_net_weights(self, model_input):
        instance_idx = model_input["instance_idx"]
        embedding = self.latent_codes(instance_idx)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def get_latent_code(self, instance_idx):
        embedding = self.latent_codes(instance_idx)
        return embedding

    # def get_latent_code(self, points):
    # latent = self.encoder(points)
    # if self.train:
    # self.z_mu = latent[..., : self.latent_dim]
    # self.z_var = latent[..., self.latent_dim :]
    # std = torch.exp(self.z_var / 2)
    # eps = torch.randn_like(std)
    # latent = eps.mul(std).add_(self.z_mu)
    # embedding = latent[..., : self.latent_dim]
    # return embedding

    # for generation
    def inference(self, coords, embedding):
        with torch.no_grad():
            model_in = {"coords": coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)

            deformation = model_output["model_out"][:, :, :3]
            correction = model_output["model_out"][:, :, 3:]
            new_coords = coords + deformation
            model_input_temp = {"coords": new_coords}
            model_output_temp = self.template_field(model_input_temp)
            return model_output_temp["model_out"] + correction

    def get_template_coords(self, coords, embedding):
        with torch.no_grad():
            model_in = {"coords": coords}
            hypo_params = self.hyper_net(embedding)
            model_output = self.deform_net(model_in, params=hypo_params)
            deformation = model_output["model_out"][:, :, :3]
            new_coords = coords + deformation

            return new_coords

    def get_template_field(self, coords):
        with torch.no_grad():
            model_in = {"coords": coords}
            model_output = self.template_field(model_in)

            return model_output["model_out"]

    # for training
    def forward(self, model_input, gt, embedding=None, **kwargs):
        instance_idx = model_input["instance_idx"]
        coords = model_input["coords"]  # 3 dimensional input coordinates

        embedding = self.get_latent_code(instance_idx)

        # get network weights for Deform-net using Hyper-net
        hypo_params = self.hyper_net(embedding)

        # [deformation field, correction field]
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output["model_out"][
            :, :, :3
        ]  # 3 dimensional deformation field
        correction = model_output["model_out"][:, :, 3:]  # scalar correction field
        new_coords = coords + deformation  # deform into template space

        # calculate gradient of the deformation field
        x = model_output["model_in"]  # input coordinates
        u = deformation[:, :, 0]
        v = deformation[:, :, 1]
        w = deformation[:, :, 2]

        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(
            u, [x], grad_outputs=grad_outputs, create_graph=True
        )[0]
        grad_v = torch.autograd.grad(
            v, [x], grad_outputs=grad_outputs, create_graph=True
        )[0]
        grad_w = torch.autograd.grad(
            w, [x], grad_outputs=grad_outputs, create_graph=True
        )[0]
        grad_deform = torch.stack(
            [grad_u, grad_v, grad_w], dim=2
        )  # gradient of deformation wrt. input position

        model_input_temp = {"coords": new_coords}

        model_output_temp = self.template_field(model_input_temp)

        sdf = model_output_temp["model_out"]  # SDF value in template space
        grad_temp = torch.autograd.grad(
            sdf, [new_coords], grad_outputs=torch.ones_like(sdf), create_graph=True
        )[
            0
        ]  # normal direction in template space

        sdf_final = sdf + correction  # add correction

        grad_sdf = torch.autograd.grad(
            sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True
        )[
            0
        ]  # normal direction in original shape space

        model_out = {
            "model_in": model_output["model_in"],
            "grad_temp": grad_temp,
            "grad_deform": grad_deform,
            "model_out": sdf_final,
            "latent_vec": embedding,
            "hypo_params": hypo_params,
            "grad_sdf": grad_sdf,
            "sdf_correct": correction,
            # "z_mu": self.z_mu,
            # "z_var": self.z_var,
        }
        losses = deform_implicit_loss(
            model_out,
            gt,
            loss_grad_deform=kwargs["loss_grad_deform"],
            loss_grad_temp=kwargs["loss_grad_temp"],
            loss_correct=kwargs["loss_correct"],
        )

        return losses

    # for evaluation
    def embedding(self, embed, model_input, gt):
        coords = model_input["coords"]  # 3 dimensional input coordinates

        # get network weights for Deform-net using Hyper-net
        hypo_params = self.hyper_net(embed)

        # [deformation field, correction field]
        model_output = self.deform_net(model_input, params=hypo_params)

        deformation = model_output["model_out"][
            :, :, :3
        ]  # 3 dimensional deformation field
        correction = model_output["model_out"][:, :, 3:]  # scalar correction field
        new_coords = coords + deformation  # deform into template space

        model_input_temp = {"coords": new_coords}

        model_output_temp = self.template_field(model_input_temp)

        sdf = model_output_temp["model_out"]  # SDF value in template space
        sdf_final = sdf + correction  # add correction

        x = model_output["model_in"]  # input coordinates
        grad_sdf = torch.autograd.grad(
            sdf_final, [x], grad_outputs=torch.ones_like(sdf), create_graph=True
        )[
            0
        ]  # normal direction in original shape space

        model_out = {
            "model_in": model_output["model_in"],
            "model_out": sdf_final,
            "latent_vec": embed,
            "grad_sdf": grad_sdf,
        }
        losses = embedding_loss(model_out, gt)

        return losses

    # Make encoder replicate the learned latent space
    def fit_encoder(self, embed, model_input, gt):
        instance_idx = model_input["instance_idx"]
        embedding = self.get_latent_code(instance_idx)
        model_out = {
            "latent_vec": embed,
            "gt_latent_vec": embedding,
        }
        losses = encoder_loss(model_out, gt)
        return losses
