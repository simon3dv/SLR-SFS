# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from models.losses.ssim import ssim
from models.networks.architectures import VGG19

import torch.nn.functional as F

class MotionLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Get the losses
        print(opt.motion_losses)
        print(zip(*[l.split("_") for l in opt.motion_losses]))
        lambdas, loss_names = zip(*[l.split("_") for l in opt.motion_losses])
        lambdas = [float(l) for l in lambdas]
        #loss_names += ("PSNR", "SSIM")

        self.lambdas = lambdas
        self.loss_names = loss_names
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "MotionL1":
            loss = MotionL1LossWrapper()
        elif name == "EndPointError":
            loss = MotionEnePointErrorWrapper()

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_motion, gt_motion, image=None, mask=None):
        losses = []
        for i, loss_name in enumerate(self.loss_names):
            if ("EP" not in loss_name) and ("Reconstruction" not in loss_name):
                losses.append(self.losses[i](pred_motion, gt_motion))
            else:
                losses.append(self.losses[i](pred_motion, gt_motion, image))

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                        loss_dir["Total Loss"]
                        + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"] * self.lambdas[i]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l
        return loss_dir


class SynthesisLoss(nn.Module):
    def __init__(self, opt, subname=""):
        super().__init__()
        self.opt = opt
        self.subname = subname
        # Get the losses
        print(opt.losses)
        print(zip(*[l.split("_") for l in opt.losses]))
        lambdas, loss_names = zip(*[l.split("_") for l in opt.losses])
        lambdas = [float(l) for l in lambdas]

        loss_names += ("PSNR", "SSIM")

        self.lambdas = lambdas
        self.losses = nn.ModuleList(
            [self.get_loss_from_name(loss_name) for loss_name in loss_names]
        )

    def get_loss_from_name(self, name):
        if name == "l1":
            loss = L1LossWrapper(self.subname)
        elif name == "content":
            loss = PerceptualLoss(self.opt, self.subname)
        elif name == "PSNR":
            loss = PSNR(self.subname)
        elif name == "SSIM":
            loss = SSIM(self.subname)
        elif name == "style":
            loss = StyleLoss(self.opt, self.subname)
        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_img, gt_img):
        losses = [loss(pred_img, gt_img) for loss in self.losses]

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                        loss_dir["Total Loss"]
                        + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l

        return loss_dir



class PSNR(nn.Module):
    def __init__(self, subname=""):
        super().__init__()
        self.subname = subname
    def forward(self, pred_img, gt_img):
        bs = pred_img.size(0)
        mse_err = (pred_img - gt_img).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)

        psnr = 10 * (1 / mse_err).log10()
        return {"psnr"+self.subname: psnr.mean()}


class SSIM(nn.Module):
    def __init__(self, subname=""):
        super().__init__()
        self.subname = subname
    def forward(self, pred_img, gt_img):
        return {"ssim"+self.subname: ssim(pred_img, gt_img)}


# Wrapper of the L1Loss so that the format matches what is expected
class L1LossWrapper(nn.Module):
    def __init__(self, subname=""):
        super().__init__()
        self.subname = subname
    def forward(self, pred_img, gt_img):
        err = nn.L1Loss()(pred_img, gt_img)
        return {"L1" + self.subname: err, "Total Loss": err}

class MotionL1LossWrapper(nn.Module):
    def forward(self, pred_motion, gt_motion):
        err = nn.L1Loss()(pred_motion, gt_motion)
        return {"MotionL1": err, "Total Loss": err}

class MotionEnePointErrorWrapper(nn.Module):
    def forward(self, pred_motion, gt_motion):
        if pred_motion.shape[1] == 3:
            new_pred_motion = pred_motion[:,:2,:,:] * pred_motion[:,2:3,:,:]
        else:
            new_pred_motion = pred_motion
        if gt_motion.shape[1] == 3:
            new_gt_motion = gt_motion[:,:2,:,:] * gt_motion[:,2:3,:,:]
        else:
            new_gt_motion = gt_motion
        #err = nn.MSELoss(reduction='none')(new_pred_motion, new_gt_motion).sum(1).sqrt()
        err = torch.norm(new_pred_motion - new_gt_motion, 2, 1)
        err = err.mean()
        return {"EndPointError": err, "Total Loss": err}

# Implementation of the perceptual loss to enforce that a
# generated image matches the given image.
# Adapted from SPADE's implementation
# (https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py)
class PerceptualLoss(nn.Module):
    def __init__(self, opt, subname=""):
        super().__init__()
        self.model = VGG19(
            requires_grad=False
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.subname = subname

    def forward(self, pred_img, gt_img, name = 'Perceptual'):
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)
        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return {"Perceptual" + self.subname : loss, "Total Loss": loss}

class StyleLoss(nn.Module):
    def __init__(self, opt, subname=""):
        super().__init__()
        self.model = VGG19(
            requires_grad=False
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.subname = subname
        self.mse_loss = nn.MSELoss()

    def gram_matrix(self, input_tensor):
        """
        Compute Gram matrix
        :param input_tensor: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: Gram matrix of y
        """
        (b, ch, h, w) = input_tensor.size()
        features = input_tensor.view(b, ch, w * h)
        features_t = features.transpose(1, 2)

        # more efficient and formal way to avoid underflow for mixed precision training
        input = torch.zeros(b, ch, ch).type(features.type())
        gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1. / (ch * h * w), out=None)

        # naive way to avoid underflow for mixed precision training
        # features = features / (ch * h)
        # gram = features.bmm(features_t) / w

        # for fp32 training, it is also safe to use the following:
        # gram = features.bmm(features_t) / (ch * h * w)

        return gram

    def forward(self, pred_img, gt_img, name = 'Perceptual'):
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)
        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        style_loss = 0
        for i in range(0, len(gt_fs)):
            #loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])
            gm_pred = self.gram_matrix(pred_fs[i])
            gm_gt = self.gram_matrix(gt_fs[i])
            style_loss += self.weights[i] * self.mse_loss(gm_pred, gm_gt.detach())
        return {"Style" + self.subname : style_loss, "Total Loss": style_loss}

