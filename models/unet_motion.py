# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.synthesis import SynthesisLoss
from models.losses.synthesis import MotionLoss
from models.networks.architectures import Unet4Motion,SPADEUnet4MaskMotion,SPADEUnet4Motion
from models.networks.utilities import get_motion_SPADE
from models.networks.utilities import get_decoder, get_encoder

import time
from utils.utils import AverageMeter
import numpy as np
import ipdb
import copy
from models.losses.synthesis import PerceptualLoss
import cv2

import time
from models import softsplat

from models.networks.architectures import ResNetEncoder
import torch.nn.functional as F
from models.losses.synthesis import PSNR
DEBUG = False
DEBUG_LOSS = False

class UnetMotion(nn.Module):
	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		self.div_flow = 20.0 if "div_flow" not in self.opt else self.opt.div_flow
		# ENCODER
		# Encode features to a given resolution
		channels_in = 3
		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			channels_in += 1
		if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
			channels_in += 2

		self.motion_predictor = Unet4Motion(channels_in=channels_in, channels_out=2, opt=opt)

		# LOSS FUNCTION
		# Module to abstract away the loss function complexity
		self.loss_function = MotionLoss(opt=opt)
		self.motion_psnr = PSNR()
	def forward(self, batch):
		""" Forward pass of a view synthesis model with a voxel latent field.
		"""
		# Input values
		image = batch["images"][0] # B, 3, W, W
		bs = image.shape[0]
		gt_motion = batch["motions"] # B, 2, W, W
		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			motion_speed = (gt_motion[:, 0:1,:] ** 2 + gt_motion[:, 1:2,:] ** 2).sqrt().view(bs, 1, self.opt.W, self.opt.W)
			# motion_speed_norm = motion_speed / motion_speed.max()
			small_motion_alpha = (motion_speed < motion_speed.mean([1,2,3],True)*0.1).float()
			moving_region_mask = 1.0 - small_motion_alpha
		if len(gt_motion.shape) <= 3:
			gt_motion = gt_motion.view(bs, 2, self.opt.motionH, self.opt.motionW)
		if torch.cuda.is_available():
			image = image.cuda()
			gt_motion = gt_motion.cuda()
		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
				hint = batch["hints"].view(moving_region_mask.shape[0], 2, moving_region_mask.shape[2],moving_region_mask.shape[3])  # B, 2, W, W
				concat_input = torch.cat([image, moving_region_mask, hint], 1)
			else:
				concat_input = torch.cat([image, moving_region_mask], 1)
			pred_motion = self.motion_predictor(concat_input)
		else:
			pred_motion = self.motion_predictor(image)

		loss = self.loss_function(pred_motion * self.div_flow, gt_motion, image)
		loss['PSNR_motion'] = self.motion_psnr(pred_motion * self.div_flow, gt_motion)['psnr']
		pred_dict = {
				"PredMotion": pred_motion,
					"GTMotion": gt_motion,

					"InputImg": image,
			}
		if"use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			pred_dict["MovingMask"]= moving_region_mask
		if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
			pred_dict["HintMotion"]= hint
		return (
			loss,pred_dict

		)
	def forward_flow(self, image,gt_mask=None, gt_hint=None):
		# Input values
		bs = image.shape[0]

		if torch.cuda.is_available():
			image = image.cuda()

		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
				concat_input = torch.cat([image, gt_mask, gt_hint], 1)
			else:
				concat_input = torch.cat([image, gt_mask], 1)
			pred_motion = self.motion_predictor(concat_input) * self.div_flow
		else:
			pred_motion = self.motion_predictor(image) * self.div_flow

		return {"PredMotion": pred_motion,}

class SPADEUnetMaskMotion(nn.Module):
	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		self.div_flow = 20.0 if "div_flow" not in self.opt else self.opt.div_flow
		# ENCODER
		# Encode features to a given resolution
		channels_in = 3
		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			channels_in += 1
		if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
			channels_in += 2

		self.motion_predictor = SPADEUnet4MaskMotion(channels_in=channels_in, channels_out=2, opt=opt)

		# LOSS FUNCTION
		# Module to abstract away the loss function complexity
		self.loss_function = MotionLoss(opt=opt)
		self.motion_psnr = PSNR()
	def forward(self, batch):
		""" Forward pass of a view synthesis model with a voxel latent field.
		"""
		# Input values
		image = batch["images"][0] # B, 3, W, W
		bs = image.shape[0]
		gt_motion = batch["motions"] # B, 2, W, W
		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			motion_speed = (gt_motion[:, 0:1,:] ** 2 + gt_motion[:, 1:2,:] ** 2).sqrt().view(bs, 1, self.opt.W, self.opt.W)
			# motion_speed_norm = motion_speed / motion_speed.max()
			small_motion_alpha = (motion_speed < motion_speed.mean([1,2,3],True)*0.1).float()
			moving_region_mask = 1.0 - small_motion_alpha
		if len(gt_motion.shape) <= 3:
			gt_motion = gt_motion.view(bs, 2, self.opt.motionH, self.opt.motionW)
		if torch.cuda.is_available():
			image = image.cuda()
			gt_motion = gt_motion.cuda()
		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
				hint = batch["hints"].view(moving_region_mask.shape[0], 2, moving_region_mask.shape[2],moving_region_mask.shape[3])  # B, 2, W, W
				concat_input = torch.cat([image, moving_region_mask, hint], 1)
			else:
				concat_input = torch.cat([image, moving_region_mask], 1)
			pred_motion = self.motion_predictor(concat_input)
		else:
			pred_motion = self.motion_predictor(image)

		loss = self.loss_function(pred_motion * self.div_flow, gt_motion, image)
		loss['PSNR_motion'] = self.motion_psnr(pred_motion * self.div_flow, gt_motion)['psnr']
		pred_dict = {
				"PredMotion": pred_motion,
					"GTMotion": gt_motion,

					"InputImg": image,
			}
		if"use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			pred_dict["MovingMask"]= moving_region_mask
		if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
			pred_dict["HintMotion"]= hint
		return (
			loss,pred_dict

		)
	def forward_flow(self, image,gt_mask=None, gt_hint=None):
		# Input values
		bs = image.shape[0]

		if torch.cuda.is_available():
			image = image.cuda()

		if "use_mask_as_motion_input" in self.opt and self.opt.use_mask_as_motion_input:
			if "use_hint_as_motion_input" in self.opt and self.opt.use_hint_as_motion_input:
				concat_input = torch.cat([image, gt_mask, gt_hint], 1)
			else:
				concat_input = torch.cat([image, gt_mask], 1)
			pred_motion = self.motion_predictor(concat_input) * self.div_flow
		else:
			pred_motion = self.motion_predictor(image) * self.div_flow

		return {"PredMotion": pred_motion}

