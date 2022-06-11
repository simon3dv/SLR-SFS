import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.synthesis import SynthesisLoss
from models.networks.architectures import Unet
from models.networks.utilities import get_decoder, get_encoder, get_decoder_bg, get_net_bg, get_alpha_encoder, get_alpha_decoder
#from models.projection.animating_z_buffer_manipulator import AnimatingPtsManipulator
from models.projection.euler_integration_manipulator import EulerIntegration, euler_integration

import time
from utils.utils import AverageMeter
import numpy as np
import copy
from models.losses.synthesis import PerceptualLoss
import cv2

import time
from models import softsplat
import math

from models.networks.architectures import ResNetEncoder
from models.networks.architectures import ResNetDecoder
from models.networks.architectures import ResBlockAlphaDecoder
from models.networks.networks import define_G
from models.networks.architectures import ResNetEncoder_with_Z
from utils.utils import AverageMeter
from models.losses.synthesis import MotionLoss
from models.networks.architectures import VGG19
#from models.unet_motion import UnetMotion
#from pytorch3d.structures.meshes import Meshes
#from pytorch3d.renderer.mesh.textures import Textures
#from pytorch3d.renderer.mesh import rasterize_meshes
#from pytorch3d.renderer.cameras import FoVPerspectiveCameras, PerspectiveCameras
#from pytorch3d.renderer import (
#	RasterizationSettings, BlendParams,
#	MeshRenderer, MeshRasterizer
#)

#import pytorch3d

#from pytorch3d.renderer import (
#	look_at_view_transform,
#	FoVPerspectiveCameras,
#	PointLights,
#	DirectionalLights,
#	Materials,
#	RasterizationSettings,
#	MeshRenderer,
#	MeshRasterizer,
#	SoftPhongShader,
#	TexturesUV,
#	TexturesVertex
#)

#import quaternion
#from scipy.spatial.transform import Rotation as R
import torchvision
DEBUG = False
DEBUG_LOSS = False


def SmoothL1Loss(input, target, gamma=0.1):  # beta=1. / 9):
	t = torch.abs(input - target)
	return t + gamma * (2 * F.sigmoid(5 * t) - 1)

def total_variation_loss(image):
	# shift one pixel and get difference (for both x and y direction)
	loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
	       torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
	return loss


def softmax_rgb_blend(
    colors, fragments, blend_params, in_channel=3, znear: float = 1.0, zfar: float = 100
) -> torch.Tensor:
	N, H, W, K = fragments.pix_to_face.shape
	device = fragments.pix_to_face.device
	pixel_colors = torch.ones((N, H, W, in_channel + 1), dtype=colors.dtype, device=colors.device)
	background = blend_params.background_color
	if not torch.is_tensor(background):
		background = torch.tensor(background, dtype=torch.float32, device=device)
	else:
		background = background.to(device)

	# Weight for background color
	eps = 1e-10

	# Mask for padded pixels.
	mask = (fragments.pix_to_face >= 0).float()

	# Sigmoid probability map based on the distance of the pixel to the face.
	prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

	# The cumulative product ensures that alpha will be 0.0 if at least 1
	# face fully covers the pixel as for that face, prob will be 1.0.
	# This results in a multiplication by 0.0 because of the (1.0 - prob)
	# term. Therefore 1.0 - alpha will be 1.0.
	alpha = torch.prod((1.0 - prob_map), dim=-1)

	# Weights for each face. Adjust the exponential by the max z to prevent
	# overflow. zbuf shape (N, H, W, K), find max over K.
	# TODO: there may still be some instability in the exponent calculation.

	z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
	# pyre-fixme[16]: `Tuple` has no attribute `values`.
	# pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
	z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
	# pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
	weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

	# Also apply exp normalize trick for the background color weight.
	# Clamp to ensure delta is never 0.
	# pyre-fixme[20]: Argument `max` expected.
	# pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
	delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

	# Normalize weights.
	# weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
	denom = weights_num.sum(dim=-1)[..., None] + delta

	# Sum: weights * textures + background color
	weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
	weighted_background = delta * background
	pixel_colors[..., :in_channel] = (weighted_colors + weighted_background) / denom
	pixel_colors[..., in_channel] = 1.0 - alpha

	return pixel_colors

class AnimatingSoftmaxSplatingJoint(nn.Module):
	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		W = opt.W
		self.div_flow = 20.0 if "div_flow" not in opt else opt.div_flow
		# ENCODER
		# Encode features to a given resolution
		self.encoder = get_encoder(opt)
		# POINT CLOUD TRANSFORMER
		# REGRESS 3D POINTS
		# self.pts_regressor = Unet(channels_in=3, channels_out=1, opt=opt)
		if "train_motion" in opt and opt.train_motion:
			from options.options import get_model
			opts_model = copy.deepcopy(self.opt)
			opts_model.model_type = self.opt.motion_model_type
			self.motion_regressor = get_model(opts_model)
		if "modifier" in self.opt.depth_predictor_type:
			self.modifier = Unet(channels_in=64, channels_out=64, opt=opt)


		if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
			self.maximum_warp_norm_splater = softsplat.ModuleMaximumWarpNormsplat()
		if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
			self.splatter = self.get_splatter(
				opt.splatter, None, opt, size=W, C=opt.ngf, points_per_pixel=opt.pp_pixel
			)
			# create xyzs(1,3,256x256) tensor for image
			xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
			ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

			xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
			ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

			xyzs = torch.cat(
				(xs, ys, torch.ones(xs.size())), 1
			).view(1, 3, -1)  # torch.Size([1, 4, 65536])
			self.register_buffer("xyzs", xyzs)
		self.euler_integration = EulerIntegration(self.opt)
		strType = 'summation'
		self.softsplater = softsplat.ModuleSoftsplat(strType)
		if "train_bg" in self.opt and self.opt.train_bg:
			self.net_bg = get_net_bg(opt)
		if "train_alpha" in self.opt and self.opt.train_alpha:
			self.net_alpha_encoder = get_alpha_encoder(opt)
			self.net_alpha_decoder = get_alpha_decoder(opt)
		self.projector = get_decoder(opt)  # UNetDecoder16Pconvnobnnonorm(opt, channels_in=65, channels_out=4)

		# LOSS FUNCTION
		# Module to abstract away the loss function complexity
		self.loss_function = SynthesisLoss(opt=opt)

		self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
		self.max_tensor = self.register_buffer(
			"max_z", torch.Tensor([self.opt.max_z])
		)
		self.discretized = self.register_buffer(
			"discretized_zs",
			torch.linspace(self.opt.min_z, self.opt.max_z, self.opt.voxel_size),
		)


	# if "train_style" in self.opt and self.opt.train_style:
	# self.loss_style =

	def get_splatter(
			self, name, depth_values, opt=None, size=256, C=64, points_per_pixel=8
	):
		if name == "xyblending":
			from models.layers.z_buffer_layers import RasterizePointsXYsBlending

			return RasterizePointsXYsBlending(
				C,
				learn_feature=opt.learn_default_feature,  # False
				radius=opt.radius,
				size=size,
				points_per_pixel=points_per_pixel,
				opts=opt,
			)
		elif name == "xyblending":
			from models.layers.z_buffer_layers import RasterizePointsXYAlphasBlending

			return RasterizePointsXYAlphasBlending(
				C,
				learn_feature=opt.learn_default_feature,  # False
				radius=opt.radius,
				size=size,
				points_per_pixel=points_per_pixel,
				opts=opt,
			)
		else:
			raise NotImplementedError()

	def random_ff_mask(self):
		"""Generate a random free form mask with configuration.
		Args:
			config: Config should have configuration including IMG_SHAPES,
				VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
		Returns:
			tuple: (top, left, height, width)
		"""

		h = self.opt.W
		w = self.opt.W
		mask = np.zeros((h, w))
		num_v = 5 + np.random.randint(
			self.opt.random_ff_mask_mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

		for i in range(num_v):
			start_x = np.random.randint(w)
			start_y = np.random.randint(h)
			for j in range(1 + np.random.randint(5)):
				angle = 0.01 + np.random.rand() * self.opt.random_ff_mask_ma
				if i % 2 == 0:
					angle = 2 * 3.1415926 - angle
				length = np.random.randint(self.opt.random_ff_mask_ml) + 1  # + 10
				brush_w = np.random.randint(self.opt.random_ff_mask_mbw) + 1  # + 10
				end_x = (start_x + length * np.sin(angle)).astype(np.int32)
				end_y = (start_y + length * np.cos(angle)).astype(np.int32)

				cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
				start_x, start_y = end_x, end_y

		return mask.reshape((1, 1,) + mask.shape).astype(np.float32)

	def forward(self, batch):
		""" Forward pass of a view synthesis model with a voxel latent field.
		"""
		# Input values
		start_img = batch["images"][0]  # B, 3, W, W
		middle_img = batch["images"][1]
		if "use_start_img_only" in self.opt and self.opt.use_start_img_only:
			end_img = batch["images"][0]
		else:
			end_img = batch["images"][2]
		mean_img = batch["mean_video"][0]

		bs = start_img.shape[0]

		if "depths" in batch.keys():
			depth_img = batch["depths"][0]
		else:
			depth_img = torch.ones(start_img[:, 0:1, :, :].shape)

		if isinstance(batch["index"], list):
			start_index = batch["index"][0]
			middle_index = batch["index"][1]
			end_index = batch["index"][2]
		elif len(batch["index"].shape) > 1:
			start_index = batch["index"][:, 0]
			middle_index = batch["index"][:, 1]
			end_index = batch["index"][:, 2]
		else:
			start_index = batch["index"][0]
			middle_index = batch["index"][1]
			end_index = batch["index"][2]

		if torch.cuda.is_available():
			start_img = start_img.cuda()
			middle_img = middle_img.cuda()
			end_img = end_img.cuda()
			mean_img = mean_img.cuda()
			if "depths" in batch.keys():
				depth_img = depth_img.cuda()

		# flow_f = flow_f.cuda()
		# flow_p = flow_p.cuda()

		# Regreesed Motion
		if "train_motion" in self.opt and self.opt.train_motion:
			flow_gt = batch["motions"]
			batch4motion = {
				"images": [start_img],
				"motions": batch["motions"],
			}
			if "hints" in batch.keys():
				batch4motion['hints'] = batch['hints']
			motion_loss, outputs_motion = self.motion_regressor.forward(
				batch4motion)
			flow_uvm = outputs_motion["PredMotion"]
			if flow_uvm.shape[1] == 2:
				flow = flow_uvm.view(bs, 2, self.opt.W, self.opt.W)
				use_uvm = False
			elif flow_uvm.shape[1] == 3:
				flow = flow_uvm[:, :2, ...] * flow_uvm[:, 2:3, ...]
				flow = flow.view(bs, 2, self.opt.W,self.opt.W)
				use_uvm = True
			else:
				print("Not Implemented")
			flow_scale = [self.opt.W / 256, self.opt.W / 256]
			flow = flow * torch.FloatTensor(flow_scale).view(1, 2, 1, 1).to(flow.device)  # train in W256
			flow = flow.view(bs,2,-1)
		else:
			if batch["motions"].shape[1] == 2:
				flow = batch["motions"].view(bs, 2, -1).cuda()
				use_uvm = False
			elif batch["motions"].shape[1] == 3:
				use_uvm = True
				flow_uvm = batch["motions"]
				flow = batch["motions"][:, :2, ...].clone()
				flow = flow * batch["motions"][:, 2:3, ...]
				flow = flow.view(bs, 2, -1).cuda()

		motion_speed = (flow[:, 0:1, :] ** 2 + flow[:, 1:2, :] ** 2).sqrt().view(bs, 1, self.opt.W, self.opt.W)
		# motion_speed_norm = motion_speed / motion_speed.max()
		if "use_MotionMaskThreshold" in self.opt and self.opt.use_MotionMaskThreshold:
			small_motion_alpha = (motion_speed < self.opt.MotionMaskThreshold).float()
		else:
			small_motion_alpha = (motion_speed < motion_speed.mean([1, 2, 3], True) * 0.1).float()

		mask_rock = batch["mask_rock"].cuda()
		moving_region_mask = 1.0-small_motion_alpha
		label = mask_rock.float() * moving_region_mask + (1 - moving_region_mask) * (
			2.0)  # 1: moving region rock 2: static region 0:moving region fluid
		if self.opt.use_rgb_features:
			pass
		else:
			ngf = self.opt.ngf
			start_fs = self.encoder(start_img)  # 2,out_channel,W,W
			end_fs = self.encoder(end_img)
			if "train_inpaint" in self.opt and self.opt.train_inpaint:
				middle_fs = self.encoder(middle_img)

			# Fluid Encoder Network
			if self.opt.use_rgb_features:
				start_fs = start_img
				end_fs = end_img
			else:
				start_fs = self.encoder(start_img)
				end_fs = self.encoder(end_img)
			if len(start_fs) == 3:
				start_fs, Z_f, _ = start_fs
				end_fs, Z_p, _ = end_fs
			elif "train_Z" in self.opt and self.opt.train_Z and not self.opt.use_rgb_features:
				start_fs, Z_f = start_fs
				end_fs, Z_p = end_fs

			# Background Network
			gen_bg_img_f = self.net_bg(start_img)
			if "use_bg_as_alpha_input" in self.opt and self.opt.use_bg_as_alpha_input:
				gen_bg_img_f_raw = gen_bg_img_f.clone()
			gen_bg_img_f = nn.Tanh()(gen_bg_img_f)


			start_alpha_input = start_img
			end_alpha_input = end_img
			if "use_motion_as_alpha_input" in self.opt and self.opt.use_motion_as_alpha_input:
				start_alpha_input = torch.cat([start_alpha_input, flow.view(bs, 2, self.opt.W, self.opt.W)], 1)
				end_alpha_input = torch.cat([end_alpha_input, flow.view(bs, 2, self.opt.W, self.opt.W)], 1)
			if "use_mask_as_alpha_input" in self.opt and self.opt.use_mask_as_alpha_input:
				start_alpha_input = torch.cat([start_alpha_input, mask_rock.view(bs, 1, self.opt.W, self.opt.W)], 1)
				end_alpha_input = torch.cat([end_alpha_input, mask_rock.view(bs, 1, self.opt.W, self.opt.W)], 1)
			if "use_bg_as_alpha_input" in self.opt and self.opt.use_bg_as_alpha_input:
				start_alpha_input = torch.cat([start_alpha_input, gen_bg_img_f_raw.view(bs, 3, self.opt.W, self.opt.W)], 1)
				end_alpha_input = torch.cat([end_alpha_input, gen_bg_img_f_raw.view(bs, 3, self.opt.W, self.opt.W)], 1)
			# Alpha Network
			alpha_output = self.net_alpha_encoder(start_alpha_input)

			if "use_sum1_alpha" in self.opt and self.opt.use_sum1_alpha:
				alpha_fluid_f = alpha_output[:, 0:1, :, :]
				alpha_bg_f = 1.0 - nn.Sigmoid()(alpha_fluid_f)
			else:
				alpha_bg_f = alpha_output[:, 0:1, :, :]
				alpha_fluid_f = alpha_output[:, 1:2, :, :]
				alpha_bg_f_raw = alpha_bg_f.clone()
				#alpha_fluid_f_raw = alpha_fluid_f.clone()
				alpha_bg_f = nn.Sigmoid()(alpha_bg_f)


			alpha_output = self.net_alpha_encoder(end_alpha_input)
			if "use_sum1_alpha" in self.opt and self.opt.use_sum1_alpha:
				alpha_fluid_p = alpha_output[:, 0:1, :, :]
				alpha_bg_p = 1.0 - nn.Sigmoid()(alpha_fluid_p)
			else:
				alpha_bg_p = alpha_output[:, 0:1, :, :]
				alpha_fluid_p = alpha_output[:, 1:2, :, :]
				#alpha_bg_p_raw = alpha_bg_p.clone()
				#alpha_fluid_p_raw = alpha_fluid_p.clone()
				alpha_bg_p = nn.Sigmoid()(alpha_bg_p)

			if self.opt.AKLloss > 0:
				alpha_logsigma = alpha_output[:, 2:3, :, :]
				alpha_logsigma = torch.clamp(alpha_logsigma, min=-50, max=50)

		if self.opt.use_gt_mean_img:
			gen_bg_img_f = mean_img.clone()
		elif self.opt.use_input_img:
			gen_bg_img_f = start_img.clone()

		alpha_0_norm = torch.clamp((nn.Sigmoid()(alpha_fluid_f) + alpha_bg_f), min=1e-8)
		CompositeFluidAlpha_I0 = (nn.Sigmoid()(alpha_fluid_f)) / alpha_0_norm

		if "use_fluid_alpha_only" in self.opt and self.opt.use_fluid_alpha_only:
			CompositeFluidAlpha_I0 = nn.Sigmoid()(alpha_fluid_f)
		if "use_bg_alpha_only" in self.opt and self.opt.use_bg_alpha_only:
			CompositeFluidAlpha_I0 = alpha_bg_f

		if "use_alpha_softmax" in self.opt and self.opt.use_alpha_softmax > 0:
			CompositeFluidAlpha_I0 = torch.cat([alpha_fluid_f, alpha_bg_f_raw], 1)
			CompositeFluidAlpha_I0 = F.softmax(CompositeFluidAlpha_I0, dim=1)[:, :1, ...]

		# Regressed points
		if "depth" in batch.keys():
			if not (self.opt.use_gt_depth):
				if not ('use_inverse_depth' in self.opt) or not (self.opt.use_inverse_depth):
					regressed_pts = (
							nn.Sigmoid()(self.pts_regressor(start_img))
							* (self.opt.max_z - self.opt.min_z)
							+ self.opt.min_z
					)
				else:  # Kitti
					# Use the inverse for datasets with landscapes, where there
					# is a long tail on the depth distribution
					depth = self.pts_regressor(start_img)  # torch.Size([1, 1, 256, 256])
					regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 1. / 256)
			else:
				regressed_pts = depth_img
		else:
			regressed_pts = depth_img
		regressed_pts = regressed_pts.view(bs, 1, -1).cuda()

		# Euler Integration
		flow = flow.view(bs, 2, self.opt.W, self.opt.W).contiguous()
		flow_f = self.euler_integration(flow, middle_index.long() - start_index.long())
		flow_p = self.euler_integration(-flow, end_index.long() + 1 - middle_index.long())
		# linear
		# flow_f = flow * (middle_index.long() - start_index.long() - 1).view(2,1,1,1)
		# flow_p = -flow * ( end_index.long() + 1 - middle_index.long() - 1).view(2,1,1,1)
		alpha = (1.0 - (middle_index.float() - start_index.float()).float() / (
				end_index.float() - start_index.float() + 1).float()).view(bs, 1, 1, 1).cuda()
		alpha = torch.clamp(alpha , min=1.0/600.0, max=599.0/600.0)

		# forward
		if ("train_Z" not in self.opt) or (not self.opt.train_Z):
			Z_f = start_fs.new_ones([start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3]])  # B, 1, W, W
		Z_f = Z_f.view(bs, 1, self.opt.W, self.opt.W)

		if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
			Z_f_max = self.maximum_warp_norm_splater(tenInput=Z_f.contiguous().detach().clone(), tenFlow=flow_f, )
			Z_f_norm = Z_f - Z_f_max
		elif "use_softmax_splatter_v1" in self.opt and self.opt.use_softmax_splatter_v1:
			Z_f_norm = Z_f
		else:
			Z_f_norm = Z_f - Z_f.max()

		if "no_clamp_Z" in self.opt and self.opt.no_clamp_Z:
			pass
		else:
			Z_f_norm = torch.clamp(Z_f_norm, min=-20.0, max=20.0)
		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			tenInput_f = torch.cat(
					[start_fs * Z_f_norm.exp() * alpha,
					 alpha_fluid_f * CompositeFluidAlpha_I0.exp() * alpha,
					 CompositeFluidAlpha_I0.exp() * alpha,
					 Z_f_norm.exp() *  alpha ],
					1)
		else:
			tenInput_f = torch.cat(
				[start_fs * Z_f_norm.exp() * alpha, alpha_fluid_f * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha],
				1)
		if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
			forward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
			                                                                                  self.opt.W)  # 1 3 W W
			forward_pts3D[:, :2, :, :] += flow_f / (self.opt.W / 2)  # forward_flow: 1, 2 W W
			forward_pts3D = forward_pts3D.view(bs, 3, -1)
			forward_pts3D = forward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
			gen_fs_f = self.splatter(forward_pts3D,
			                         tenInput_f.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
		else:
			gen_fs_f = self.softsplater(tenInput=tenInput_f,
			                            tenFlow=flow_f,
			                            tenMetric=start_fs.new_ones(
				                            (start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3])))  # B, 65, W, W

		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			gen_fs = gen_fs_f[:, :-3, :, :]
			alpha_fluid = gen_fs_f[:, -3:-2, :, :]
			alpha_tenNormalize = gen_fs_f[:, -2:-1, :, :]
			tenNormalize = gen_fs_f[:, -1:, :, :]
		else:
			gen_fs = gen_fs_f[:, :-2, :, :]
			alpha_fluid = gen_fs_f[:, -2:-1, :, :]
			tenNormalize = gen_fs_f[:, -1:, :, :]
		# backward
		if ("train_Z" not in self.opt) or (not self.opt.train_Z):
			Z_p = start_fs.new_ones([end_fs.shape[0], 1, end_fs.shape[2], end_fs.shape[3]])
		Z_p = Z_p.view(bs, 1, self.opt.W, self.opt.W)

		if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
			Z_p_max = self.maximum_warp_norm_splater(tenInput=Z_p.contiguous().detach().clone(), tenFlow=flow_p, )
			Z_p_norm = Z_p - Z_p_max
		elif "use_softmax_splatter_v1" in self.opt and self.opt.use_softmax_splatter_v1:
			Z_p_norm = Z_p
		else:
			Z_p_norm = Z_p - Z_p.max()
		if "no_clamp_Z" in self.opt and self.opt.no_clamp_Z:
			pass
		else:
			Z_p_norm = torch.clamp(Z_p_norm, min=-20.0, max=20.0)
		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			if "use_alpha_tau" in self.opt and self.opt.use_alpha_tau:
				tenInput_p = torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha),
				                        alpha_fluid_p * CompositeFluidAlpha_I0**1.5 * (1 - alpha),
				                        CompositeFluidAlpha_I0**1.5 * (1 - alpha),
				                        Z_p_norm.exp() * (1 - alpha)], 1)
			else:
				tenInput_p = torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha),
				                        alpha_fluid_p * CompositeFluidAlpha_I0.exp() * (1 - alpha),
				                        CompositeFluidAlpha_I0.exp() * (1 - alpha),
				                        Z_p_norm.exp() *(1 - alpha)], 1)
		else:
			tenInput_p = torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha), alpha_fluid_p * Z_p_norm.exp() * (1 - alpha),
		                        Z_p_norm.exp() * (1 - alpha)], 1)

		if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
			backward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
			                                                                                   self.opt.W)  # 1 3 W W
			backward_pts3D[:, :2, :, :] += flow_p / (self.opt.W / 2)
			backward_pts3D = backward_pts3D.view(bs, 3, -1)
			backward_pts3D = backward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
			gen_fs_p = self.splatter(backward_pts3D,
			                         tenInput_p.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
		else:
			gen_fs_p = self.softsplater(tenInput=tenInput_p,
			                            tenFlow=flow_p,
			                            tenMetric=start_fs.new_ones(
				                            (start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3])))

		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			gen_fs += gen_fs_p[:, :-3, :, :]
			alpha_fluid += gen_fs_p[:, -3:-2, :, :]
			alpha_tenNormalize += gen_fs_p[:, -2:-1, :, :]
			tenNormalize += gen_fs_p[:, -1:, :, :]
		else:
			gen_fs += gen_fs_p[:, :-2, :, :]
			alpha_fluid += gen_fs_p[:, -2:-1, :, :]
			tenNormalize += gen_fs_p[:, -1:, :, :]

		alpha_fluid_mask = (tenNormalize > 1e-8).float().detach()  # (tenNormalize != 0.0).float()
		tenNormalize = torch.clamp(tenNormalize, min=1e-8)  # tenNormalize[tenNormalize == 0.0] = 1.0
		gen_fs = gen_fs / tenNormalize
		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			alpha_tenNormalize = torch.clamp(alpha_tenNormalize, min=1e-8)  # tenNormalize[tenNormalize == 0.0] = 1.0
			alpha_fluid = alpha_fluid / alpha_tenNormalize
		else:
			alpha_fluid = alpha_fluid / tenNormalize

		if "augmented_edge_mask" in self.opt and self.opt.augmented_edge_mask:
			if np.random.rand(1) < self.augmented_edge_mask_rate:
				valid_mask_cur = (gen_fs != 0).float()
				valid_mask_cur = (valid_mask_cur.sum(1, True) > 0).float()
				weight_maskUpdater = torch.ones(1, 1, self.opt.augmented_edge_mask_length,
				                                self.opt.augmented_edge_mask_length).cuda()
				F.conv2d(valid_mask_cur, weight_maskUpdater, bias=None, stride=1, padding=1)

		if "random_ff_mask" in self.opt and self.opt.random_ff_mask:
			random_ff_mask_flag = False
			if np.random.rand(1) < self.opt.random_ff_mask_rate:
				random_ff_mask_flag = True
				generated_mask = torch.FloatTensor(self.random_ff_mask())
				generated_mask = 1.0 - generated_mask
				generated_mask = generated_mask.repeat(gen_fs.shape[0], 1, 1, 1)
				generated_mask = generated_mask.to(gen_fs.device)
				gen_fs = gen_fs * generated_mask

		# Fluid Img Decoder
		gen_fluid_img = self.projector(gen_fs)
		gen_fluid_img = nn.Tanh()(gen_fluid_img)

		# Fluid Alpha Decoder
		if "decouple" in self.opt.alpha_refine_model_type:
			gen_fluid_alpha = self.net_alpha_decoder(alpha_fluid)
		elif "image" in self.opt.alpha_refine_model_type:
			gen_fluid_alpha = self.net_alpha_decoder(torch.cat([start_img, alpha_fluid],1))
		else:
			gen_fluid_alpha = self.net_alpha_decoder(torch.cat([gen_fs, alpha_fluid],1))
		gen_fluid_alpha_raw = gen_fluid_alpha.clone()
		gen_fluid_alpha = nn.Sigmoid()(gen_fluid_alpha)
		"""
		if self.opt.AKLloss > 0 and ("isval" in batch.keys() and not batch["isval"][0]):
			z_fluid = torch.randn(bs, 1, self.opt.W, self.opt.W).cuda()
			gen_fluid_alpha = nn.Sigmoid()((logsigma_fluid * 0.5).exp() * z_fluid + gen_fluid_img[:, 3:4, :, :])
		else:
			gen_fluid_alpha = nn.Sigmoid()(gen_fluid_img[:, 3:4, :, :])
		"""
		alpha_norm = gen_fluid_alpha + alpha_bg_f
		alpha_norm = torch.clamp(alpha_norm, min=1e-8)

		GTAlpha = mask_rock *  (1.0-small_motion_alpha) * 0.25 + \
		          (1.0-mask_rock) * (1.0 - small_motion_alpha) * 1 + \
		          small_motion_alpha * 0.5

		if "use_fluid_img_at_static_region" in self.opt and self.opt.use_fluid_img_at_static_region:

			alpha_bg_f = alpha_bg_f * (1 - small_motion_alpha) + (
						torch.zeros(alpha_bg_f.shape).cuda()) * small_motion_alpha
			gen_fluid_alpha = gen_fluid_alpha * (1 - small_motion_alpha) + (
				torch.zeros(gen_fluid_alpha.shape).cuda()+1) * small_motion_alpha

			gen_img = small_motion_alpha * gen_fluid_img + (1-small_motion_alpha) * \
				(gen_fluid_alpha * gen_fluid_img + alpha_bg_f * gen_bg_img_f) / alpha_norm
		elif "use_start_img_at_static_region" in self.opt and self.opt.use_start_img_at_static_region:
			alpha_bg_f = alpha_bg_f * (1 - small_motion_alpha) + (
						torch.zeros(alpha_bg_f.shape).cuda()) * small_motion_alpha
			gen_fluid_alpha = gen_fluid_alpha * (1 - small_motion_alpha) + (
				torch.zeros(gen_fluid_alpha.shape).cuda()+1) * small_motion_alpha

			gen_img = small_motion_alpha * start_img + (1-small_motion_alpha) * \
				(gen_fluid_alpha * gen_fluid_img + alpha_bg_f * gen_bg_img_f) / alpha_norm
		else:
			if "use_alpha_softmax" in self.opt and self.opt.use_alpha_softmax > 0:
				concat_alpha = torch.cat([gen_fluid_alpha_raw, alpha_bg_f_raw], 1)
				concat_alpha = F.softmax(concat_alpha, dim=1)
				gen_img = concat_alpha[:, :1, ...] * gen_fluid_img + \
				                         concat_alpha[:, 1:2, ...] * gen_bg_img_f
			elif "clamp_alpha" in self.opt and self.opt.clamp_alpha > 0:
				composited_fluid_alpha = gen_fluid_alpha / alpha_norm
				composited_fluid_alpha = torch.clamp(composited_fluid_alpha, min=self.opt.clamp_alpha)
				composited_bg_alpha = alpha_bg_f / alpha_norm
				gen_img = composited_fluid_alpha * gen_fluid_img + composited_bg_alpha * gen_bg_img_f
			else:
				gen_img = (gen_fluid_alpha * gen_fluid_img + alpha_bg_f * gen_bg_img_f) / alpha_norm

		# And the loss
		loss = self.loss_function(gen_img, middle_img)


		if "AlphaMSEloss" in self.opt and self.opt.AlphaMSEloss > 0.0:
			if 'use_reweight_mask_loss' in self.opt and self.opt.use_reweight_mask_loss:
				mask_err =  nn.MSELoss(reduction='none')(CompositeFluidAlpha_I0 * (1.0-small_motion_alpha),
				                                     GTAlpha * (1.0-small_motion_alpha))
				pos_mask = mask_rock*(1.0-small_motion_alpha)
				neg_mask = (1.0-mask_rock)*(1.0-small_motion_alpha)
				pos_mask_loss = (pos_mask * mask_err).sum() / (1 + pos_mask.sum())
				neg_mask_loss = (neg_mask * mask_err).sum() / (1 + neg_mask.sum())
				loss["AlphaMSEloss"] = .5 * (pos_mask_loss + neg_mask_loss)
				if 'use_static_moving_loss' in self.opt and self.opt.use_static_moving_loss:
					mask_bg_err = nn.MSELoss(reduction='none')(alpha_bg_f * (1.0 - small_motion_alpha),
					                                           GTAlpha * (1.0 - small_motion_alpha))
					mask_fluid_err = nn.MSELoss(reduction='none')(nn.Sigmoid()(alpha_fluid_f) * (1.0 - small_motion_alpha),
					                                              GTAlpha * (1.0 - small_motion_alpha))
					pos_mask_bg_loss = (pos_mask * mask_bg_err).sum() / (1 + pos_mask.sum())
					neg_mask_fluid_loss = (neg_mask * mask_fluid_err).sum() / (1 + neg_mask.sum())
					loss["AlphaMSEloss"] += .5 * (
							pos_mask_bg_loss + neg_mask_fluid_loss) * self.opt.AlphaMSEloss * self.opt.static_moving_weight
			elif 'use_reweight_mask_loss2' in self.opt and self.opt.use_reweight_mask_loss2:
				mask_err =  nn.MSELoss(reduction='none')(CompositeFluidAlpha_I0 * (1.0-small_motion_alpha),
				                                     GTAlpha * (1.0-small_motion_alpha))
				pos_mask = mask_rock*(1.0-small_motion_alpha)
				neg_mask = (1.0-mask_rock)*(1.0-small_motion_alpha)
				pos_mask_loss = (pos_mask * mask_err).sum() / (1 + pos_mask.sum()) * self.opt.reweight_pos
				neg_mask_loss = (neg_mask * mask_err).sum() / (1 + neg_mask.sum()) * self.opt.reweight_neg
				loss["AlphaMSEloss"] = .5 * (pos_mask_loss + neg_mask_loss)
				if 'use_static_moving_loss' in self.opt and self.opt.use_static_moving_loss:
					mask_bg_err = nn.MSELoss(reduction='none')(alpha_bg_f * (1.0 - small_motion_alpha),
					                                           GTAlpha * (1.0 - small_motion_alpha))
					mask_fluid_err = nn.MSELoss(reduction='none')(nn.Sigmoid()(alpha_fluid_f) * (1.0 - small_motion_alpha),
					                                              GTAlpha * (1.0 - small_motion_alpha))
					pos_mask_bg_loss = (pos_mask * mask_bg_err).sum() / (1 + pos_mask.sum())
					neg_mask_fluid_loss = (neg_mask * mask_fluid_err).sum() / (1 + neg_mask.sum())
					loss["AlphaMSEloss"] += .5 * (
							pos_mask_bg_loss + neg_mask_fluid_loss) * self.opt.AlphaMSEloss * self.opt.static_moving_weight
			else:
				loss["AlphaMSEloss"] = nn.MSELoss()(
					CompositeFluidAlpha_I0 * (1.0-small_motion_alpha),
					GTAlpha * (1.0-small_motion_alpha)
				).mean()

			loss['Total Loss'] += loss["AlphaMSEloss"] * self.opt.AlphaMSEloss


		if "AlphaL1loss" in self.opt and self.opt.AlphaL1loss > 0.0:
			if 'use_reweight_mask_loss' in self.opt and self.opt.use_reweight_mask_loss:
				mask_err = SmoothL1Loss(CompositeFluidAlpha_I0 * (1.0-small_motion_alpha),
				                                     GTAlpha * (1.0-small_motion_alpha))
				pos_mask = mask_rock*(1.0-small_motion_alpha)
				neg_mask = (1.0-mask_rock)*(1.0-small_motion_alpha)
				pos_mask_loss = (pos_mask * mask_err).sum() / (1 + pos_mask.sum())
				neg_mask_loss = (neg_mask * mask_err).sum() / (1 + neg_mask.sum())
				loss["AlphaL1loss"] = .5 * (pos_mask_loss + neg_mask_loss)
				if 'use_static_moving_loss' in self.opt and self.opt.use_static_moving_loss:
					mask_bg_err = SmoothL1Loss(alpha_bg_f * (1.0 - small_motion_alpha),
					                                           GTAlpha * (1.0 - small_motion_alpha))
					mask_fluid_err = SmoothL1Loss(nn.Sigmoid()(alpha_fluid_f) * (1.0 - small_motion_alpha),
					                                              GTAlpha * (1.0 - small_motion_alpha))
					pos_mask_bg_loss = (pos_mask * mask_bg_err).sum() / (1 + pos_mask.sum())
					neg_mask_fluid_loss = (neg_mask * mask_fluid_err).sum() / (1 + neg_mask.sum())
					loss["AlphaL1loss"] += .5 * (
							pos_mask_bg_loss + neg_mask_fluid_loss) * self.opt.AlphaL1loss * self.opt.static_moving_weight
			else:
				loss["AlphaL1loss"] = SmoothL1Loss(
					CompositeFluidAlpha_I0 * (1.0-small_motion_alpha),
					GTAlpha * (1.0-small_motion_alpha)
				).mean()
			loss['Total Loss'] += loss["AlphaL1loss"] * self.opt.AlphaL1loss

		if "ATVloss" in self.opt and self.opt.ATVloss>0.0:
			loss["AlphaTV"] = total_variation_loss(alpha_fluid_f) + total_variation_loss(alpha_bg_f)
			loss["Total Loss"] += loss["AlphaTV"] * self.opt.ATVloss

		if self.opt.MVloss>0.0:
			loss_bgimg = self.loss_function(gen_bg_img_f, mean_img)
			for i, loss_name in enumerate(loss_bgimg):
				if "Perceptual" in loss_name:
					loss[loss_name + "_bg"] = loss_bgimg[loss_name]
				elif "L1" in loss_name:
					loss[loss_name + "_bg"] = loss_bgimg[loss_name]
				elif "Total" in loss_name:
					loss[loss_name] += loss_bgimg[loss_name] * self.opt.MVloss #* 0.3

		if "FluidRegionloss" in self.opt and self.opt.FluidRegionloss > 0.0:
			loss["FluidRegionLoss"] = SmoothL1Loss(CompositeFluidAlpha_I0 * (1.0-mask_rock) * (1.0 - small_motion_alpha),
			                                       (torch.zeros(CompositeFluidAlpha_I0.shape).cuda() + 1.0)* (1.0-mask_rock) * (1.0 - small_motion_alpha)
			                                       ).mean()

			loss["Total Loss"] += loss["FluidRegionLoss"] * self.opt.FluidRegionloss

		if "RockRegionloss" in self.opt and self.opt.RockRegionloss > 0.0:
			mask_rock = batch["mask_rock"].cuda()
			loss["RockRegionLoss"] = \
				SmoothL1Loss(CompositeFluidAlpha_I0 * mask_rock * (1.0 - small_motion_alpha),
				             (torch.zeros(CompositeFluidAlpha_I0.shape).cuda() + self.opt.RockRegionlosstarget)* mask_rock * (1.0 - small_motion_alpha)
				             ).mean()

			loss["Total Loss"] += loss["RockRegionLoss"] * self.opt.RockRegionloss

		if self.opt.ADCloss > 0:
			loss["Alpha Decoder Consistency Loss"] = SmoothL1Loss(alpha_fluid.clone().detach() * alpha_fluid_mask,
				                                                      gen_fluid_alpha_raw * alpha_fluid_mask).mean()  # / (alpha_fluid_mask.sum()+1e-8) * self.opt.W **2 * bs
			loss["Total Loss"] += loss["Alpha Decoder Consistency Loss"] * self.opt.ADCloss

		if self.opt.MRADCloss > 0:
			loss["Moving Region Alpha Decoder Consistency Loss"] = (SmoothL1Loss(alpha_fluid.clone().detach() * alpha_fluid_mask ,
		                                                       gen_fluid_alpha_raw * alpha_fluid_mask) * (1.0 - small_motion_alpha)).mean()
			loss["Total Loss"] += loss["Moving Region Alpha Decoder Consistency Loss"] * self.opt.MRADCloss

		if "train_motion" in self.opt and self.opt.train_motion:
			lambdas, loss_names = zip(*[l.split("_") for l in self.opt.motion_losses])
			lambdas = [float(l) for l in lambdas]
			# motion_loss = self.motion_loss_function(flow, flow_gt)
			loss["Total Loss"] += motion_loss["Total Loss"]
			for i, loss_name in enumerate(loss_names):
				loss[loss_name] = motion_loss[loss_name] * lambdas[i]

		CompositeFluidAlphaNorm = torch.clamp(gen_fluid_alpha + alpha_bg_f, min=1e-8)

		pred_dict = {
			#"StartImg": start_img,
			"OutputImg": middle_img,
			#"EndImg": end_img,
			"PredImg": gen_img,
			"BGImg_f": gen_bg_img_f,
			"MeanImg": mean_img,
			"FluidImg": gen_fluid_img,
			#"WarpedAlpha": nn.Sigmoid()(alpha_fluid),
			#"RefinedAlpha": gen_fluid_alpha,
			"AlphaFluid_f": nn.Sigmoid()(alpha_fluid_f),
			"AlphaBG_f": alpha_bg_f,
			"CompositeFluidAlpha": nn.Sigmoid()(gen_fluid_alpha_raw) / CompositeFluidAlphaNorm,
			"Z_f": Z_f_norm,
			"GTMotion": flow.view(bs, 2, self.opt.W, self.opt.W),
			#"SmallSpeedAlpha": small_motion_alpha,
			#"AlphaFluidMask": alpha_fluid_mask,
			"GTAlpha": GTAlpha,
		}
		pred_dict["RockMask"] = mask_rock

		if "train_motion" in self.opt and self.opt.train_motion:
			pred_dict["PredMotion"] = outputs_motion["PredMotion"]
			pred_dict["GTMotion"] = outputs_motion["GTMotion"]

		if isinstance(pred_dict, tuple):
			pred_dict = pred_dict[0]

		for key in pred_dict.keys():
			if isinstance(pred_dict[key], tuple):
				pred_dict[key] = pred_dict[key][0]

		return loss, pred_dict

	def random_ff_mask(self):
		"""Generate a random free form mask with configuration.
		Args:
			config: Config should have configuration including IMG_SHAPES,
				VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
		Returns:
			tuple: (top, left, height, width)
		"""

		h = self.opt.W
		w = self.opt.W
		mask = np.zeros((h, w))
		num_v = 12 + np.random.randint(
			self.opt.random_ff_mask_mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

		for i in range(num_v):
			start_x = np.random.randint(w)
			start_y = np.random.randint(h)
			for j in range(1 + np.random.randint(5)):
				angle = 0.01 + np.random.randint(self.opt.random_ff_mask_ma)
				if i % 2 == 0:
					angle = 2 * 3.1415926 - angle
				length = 10 + np.random.randint(self.opt.random_ff_mask_ml)
				brush_w = 10 + np.random.randint(self.opt.random_ff_mask_mbw)
				end_x = (start_x + length * np.sin(angle)).astype(np.int32)
				end_y = (start_y + length * np.cos(angle)).astype(np.int32)

				cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
				start_x, start_y = end_x, end_y

		return mask.reshape((1,) + (1,) + mask.shape).astype(np.float32)

	def forward_flow(self, batch,vis_time=False):
		# Input values
		ngf = self.opt.ngf
		start_img = batch["images"][0]
		start_fs = batch["features"][0]
		start_fs, Z_f = start_fs
		bs = start_fs.shape[0]

		if isinstance(batch["index"], list):
			start_index = batch["index"][0]
			middle_index = batch["index"][1]
			end_index = batch["index"][2]
		elif len(batch["index"].shape) > 1:
			start_index = batch["index"][:, 0]
			middle_index = batch["index"][:, 1]
			end_index = batch["index"][:, 2]
		else:
			start_index = batch["index"][0]
			middle_index = batch["index"][1]
			end_index = batch["index"][2]

		if "mask_rock" in batch.keys():
			mask_rock = batch["mask_rock"].cuda()

		if "alpha_region" in batch.keys():
			alpha_region = batch["alpha_region"].cuda()

			kernel_size = self.opt.W // 20
			if kernel_size % 2 == 0: kernel_size = kernel_size + 1
			sigma = self.opt.W // 50

			# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
			x_cord = torch.arange(kernel_size)
			x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
			y_grid = x_grid.t()
			xy_grid = torch.stack([x_grid, y_grid], dim=-1)

			mean = (kernel_size - 1) / 2.
			variance = sigma ** 2
			# Calculate the 2-dimensional gaussian kernel which is
			# the product of two gaussian distributions for two different
			# variables (in this case called x and y)
			gaussian_kernel = (1. / (2. * math.pi * variance)) * \
			                  torch.exp(
				                  -torch.sum((xy_grid - mean) ** 2., dim=-1).float() / (2.0 * variance)
			                  )
			# Make sure sum of values in gaussian kernel equals 1.
			gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

			# Reshape to 2d depthwise convolutional weight
			gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).cuda()
			gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

			gaussian_filter = nn.Conv2d(in_channels=1,
			                                 out_channels=1,
			                                 kernel_size=kernel_size,
			                                 groups=1,
			                                 bias=False,
			                                 padding=kernel_size // 2,
			                                 padding_mode='replicate')

			gaussian_filter.weight.data = gaussian_kernel
			gaussian_filter.weight.requires_grad = False
			alpha_region = gaussian_filter(alpha_region)

		# Regreesed Motion
		if "motions" not in batch.keys():
			if "motion_model_type" in self.opt and 'resnet' in self.opt.motion_model_type:
				flow = self.MotionEncoder(start_img)
				flow = self.MotionDecoder(flow)
			else:
				flow = self.motion_regressor.forward_flow(start_img)["PredMotion"]
		elif len(batch["motions"]) == 2:
			flow_f = batch["motions"][0].cuda()
			flow_p = batch["motions"][1].cuda()
		else:
			flow = batch["motions"][0].cuda()
		# start_index, middle_index, end_index = batch['index'][0]
		forward_flow, _ = euler_integration(flow, middle_index - start_index)
		backward_flow, _ = euler_integration(-flow, end_index - middle_index + 1)

		#BG
		gen_bg_img_f = batch['BGImg'][0]#self.net_bg(start_img)
		gen_bg_img_f_raw = gen_bg_img_f.clone()
		gen_bg_img_f = nn.Tanh()(gen_bg_img_f)

		# Alpha Network
		start_alpha_input = start_img
		if "use_motion_as_alpha_input" in self.opt and self.opt.use_motion_as_alpha_input:
			start_alpha_input = torch.cat([start_alpha_input, flow.view(bs, 2, self.opt.W, self.opt.W)], 1)
		if "use_mask_as_alpha_input" in self.opt and self.opt.use_mask_as_alpha_input:
			start_alpha_input = torch.cat([start_alpha_input, mask_rock.view(bs, 1, self.opt.W, self.opt.W)], 1)
		if "use_bg_as_alpha_input" in self.opt and self.opt.use_bg_as_alpha_input:
			start_alpha_input = torch.cat([start_alpha_input, gen_bg_img_f_raw.view(bs, 3, self.opt.W, self.opt.W)], 1)

		alpha_output = self.net_alpha_encoder(start_alpha_input)
		if "use_sum1_alpha" in self.opt and self.opt.use_sum1_alpha:
			alpha_fluid_f = alpha_output[:, 0:1, :, :]
			alpha_bg_f = 1.0 - nn.Sigmoid()(alpha_fluid_f)
		else:
			alpha_bg_f = alpha_output[:, 0:1, :, :]
			alpha_fluid_f = alpha_output[:, 1:2, :, :]
			alpha_bg_f_raw = alpha_bg_f.clone()
			alpha_bg_f = nn.Sigmoid()(alpha_bg_f)


		# forward
		alpha = (1.0 - (middle_index - start_index).float() / (end_index - start_index + 1).float()).view(bs, 1, 1,
		                                                                                                  1).cuda()
		alpha = torch.clamp(alpha , min=1.0/600.0, max=599.0/600.0)


		if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
			Z_f_max = self.maximum_warp_norm_splater(tenInput=Z_f.contiguous().detach().clone(), tenFlow=forward_flow, )
			Z_f_norm = Z_f - Z_f_max
		elif "use_softmax_splatter_v1" in self.opt and self.opt.use_softmax_splatter_v1:
			Z_f_norm = Z_f
		else:
			Z_f_norm = Z_f - Z_f.max()

		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			alpha_0_norm = torch.clamp((nn.Sigmoid()(alpha_fluid_f) + alpha_bg_f), min=1e-8)
			CompositeFluidAlpha_I0 = (nn.Sigmoid()(alpha_fluid_f)) / alpha_0_norm

			tenInput_f = torch.cat(
				[start_fs * Z_f_norm.exp() * alpha,
					alpha_fluid_f * CompositeFluidAlpha_I0.exp() * alpha,
					CompositeFluidAlpha_I0.exp() * alpha,
					Z_f_norm.exp() * alpha],
					1)
		else:
			tenInput_f = torch.cat(
				[start_fs * Z_f_norm.exp() * alpha, alpha_fluid_f * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha],
				1)

		if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
			forward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
			                                                                                  self.opt.W)  # 1 3 W W
			forward_pts3D[:, :2, :, :] += forward_flow / (self.opt.W / 2)  # forward_flow: 1, 2 W W
			forward_pts3D = forward_pts3D.view(bs, 3, -1)
			forward_pts3D = forward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
			gen_fs_f = self.splatter(forward_pts3D,
		                             tenInput_f.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
		else:
			gen_fs_f = self.softsplater(tenInput=tenInput_f,  # torch.Size([bs, 65, 384, 384])
			                            tenFlow=forward_flow.view(bs, -1, self.opt.W, self.opt.W),
			                            # torch.Size([bs, 1, 384, 384])
			                            tenMetric=start_fs.new_ones((bs, 1, self.opt.W, self.opt.W)))

		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			gen_fs = gen_fs_f[:, :-3, :, :]
			alpha_fluid = gen_fs_f[:, -3:-2, :, :]
			alpha_tenNormalize = gen_fs_f[:, -2:-1, :, :]
			tenNormalize = gen_fs_f[:, -1:, :, :]
		else:
			gen_fs = gen_fs_f[:, :-2, :, :]
			alpha_fluid = gen_fs_f[:, -2:-1, :, :]
			tenNormalize = gen_fs_f[:, -1:, :, :]

		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:

			tenInput_p = torch.cat(
					[start_fs * Z_f_norm.exp() * (1 - alpha),
					 alpha_fluid_f *  CompositeFluidAlpha_I0.exp() * (1 - alpha),
					 CompositeFluidAlpha_I0.exp() * (1 - alpha) ,
					Z_f_norm.exp() * (1 - alpha)],
					1)

		else:
			tenInput_p = torch.cat([start_fs * Z_f_norm.exp() * (1 - alpha), alpha_fluid_f * Z_f_norm.exp() * (1 - alpha),
			                        Z_f_norm.exp() * (1 - alpha)], 1)

		if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
			backward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
			                                                                                   self.opt.W)  # 1 3 W W
			backward_pts3D[:, :2, :, :] += backward_flow / (self.opt.W / 2)
			backward_pts3D = backward_pts3D.view(bs, 3, -1)
			backward_pts3D = backward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
			gen_fs_p = self.splatter(backward_pts3D,
			                         tenInput_p.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
		else:
			gen_fs_p = self.softsplater(tenInput=tenInput_p,
			                            tenFlow=backward_flow.view(bs, -1, self.opt.W, self.opt.W),
			                            tenMetric=start_fs.new_ones((bs, 1, self.opt.W, self.opt.W)))

		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			gen_fs += gen_fs_p[:, :-3, :, :]
			alpha_fluid += gen_fs_p[:, -3:-2, :, :]
			alpha_tenNormalize += gen_fs_p[:, -2:-1, :, :]
			tenNormalize += gen_fs_p[:, -1:, :, :]
		else:
			gen_fs += gen_fs_p[:, :-2, :, :]
			alpha_fluid += gen_fs_p[:, -2:-1, :, :]
			tenNormalize += gen_fs_p[:, -1:, :, :]

		tenNormalize = torch.clamp(tenNormalize, min=1e-8)  # tenNormalize[tenNormalize == 0.0] = 1.0
		alpha_fluid_mask = (tenNormalize > 1e-8).float().detach()
		gen_fs = gen_fs / tenNormalize
		if "use_alpha0_as_blending_weight" in self.opt and self.opt.use_alpha0_as_blending_weight:
			alpha_tenNormalize = torch.clamp(alpha_tenNormalize, min=1e-8)  # tenNormalize[tenNormalize == 0.0] = 1.0
			alpha_fluid = alpha_fluid / alpha_tenNormalize
		else:
			alpha_fluid = alpha_fluid / tenNormalize

		# Fluid Img Decoder
		gen_fluid_img = self.projector(gen_fs)
		gen_fluid_img = nn.Tanh()(gen_fluid_img)

		# Fluid Alpha Decoder
		gen_fluid_alpha = self.net_alpha_decoder(torch.cat([gen_fs, alpha_fluid],1))
		gen_fluid_alpha_raw = gen_fluid_alpha.clone()
		gen_fluid_alpha = nn.Sigmoid()(gen_fluid_alpha)

		alpha_norm = gen_fluid_alpha + alpha_bg_f
		alpha_norm = torch.clamp(alpha_norm, min=1e-8)


		if "use_fluid_alpha_only" in self.opt and self.opt.use_fluid_alpha_only:
			alpha_norm = 1
		if "use_bg_alpha_only" in self.opt and self.opt.use_bg_alpha_only:
			alpha_norm = 1


		if "use_alpha_softmax" in self.opt and self.opt.use_alpha_softmax > 0:
			concat_alpha = torch.cat([gen_fluid_alpha_raw, alpha_bg_f_raw], 1)
			concat_alpha = F.softmax(concat_alpha, dim=1)
			gen_img = concat_alpha[:, :1, ...] * gen_fluid_img + \
			                         concat_alpha[:, 1:2, ...] * gen_bg_img_f
		elif "clamp_alpha" in self.opt and self.opt.clamp_alpha > 0:
			composited_fluid_alpha = gen_fluid_alpha / alpha_norm
			composited_fluid_alpha = torch.clamp(composited_fluid_alpha, min=self.opt.clamp_alpha)
			composited_bg_alpha = alpha_bg_f / alpha_norm
			gen_img = composited_fluid_alpha * gen_fluid_img + (1.0-composited_fluid_alpha) * gen_bg_img_f
		else:
			gen_img = (gen_fluid_alpha * gen_fluid_img + alpha_bg_f * gen_bg_img_f) / alpha_norm

		if "alpha_region" in batch.keys():
			gen_img = gen_img * alpha_region + gen_fluid_img * (1.0-alpha_region)

		if "use_fluid_alpha_only" in self.opt and self.opt.use_fluid_alpha_only:
			gen_img = gen_fluid_alpha * gen_fluid_img + (1.0-gen_fluid_alpha) * gen_bg_img_f
		if "use_bg_alpha_only" in self.opt and self.opt.use_bg_alpha_only:
			gen_img = (1.0-alpha_bg_f) * gen_fluid_img + alpha_bg_f * gen_bg_img_f


		CompositeFluidAlpha = gen_fluid_alpha / alpha_norm
		pred_dict = {"PredImg": gen_img,
		             "BGImg": gen_bg_img_f,
		             "FluidImg": gen_fluid_img,
		             #"FluidAlpha_f": nn.Sigmoid()(alpha_fluid_f),
		             "CompositeFluidAlpha": CompositeFluidAlpha,
		             #"BGAlpha_f": alpha_bg_f,
		             #"Z_f": Z_f_norm,
					 #"AlphaFluidMask": torch.clamp(alpha_fluid_mask.float(),min=0,max=1),
		             # "WarpedAlpha": nn.Sigmoid()(alpha_fluid) # Not support for visulization yet
		             # "RefinedWarpedAlpha": gen_fluid_alpha # Not support for visulization yet
		             }
		if "alpha_region" in batch.keys():
			EditCompositeFluidAlpha = CompositeFluidAlpha * alpha_region + 1.0 * (1.0-alpha_region)
			pred_dict["EditedCompositeFluidAlpha"] = EditCompositeFluidAlpha
			pred_dict["AlphaRegionMask"] = alpha_region
		if "use_fluid_alpha_only" in self.opt and self.opt.use_fluid_alpha_only:
			pred_dict["CompositeFluidAlpha"] = gen_fluid_alpha
		if "use_bg_alpha_only" in self.opt and self.opt.use_fluid_alpha_only:
			pred_dict["CompositeFluidAlpha"] = alpha_bg_f
		return pred_dict



	def forward_flow_mesh(self, batch):
		pass



class BackgroundNetwork(nn.Module):
	def __init__(self, opt):
		super().__init__()

		self.opt = opt
		W = opt.W
		self.net_bg = get_net_bg(opt)
		# LOSS FUNCTION
		# Module to abstract away the loss function complexity
		self.loss_function = SynthesisLoss(opt=opt)

	def forward(self, batch):
		""" Forward pass of a view synthesis model with a voxel latent field.
		"""
		# Input values
		start_img = batch["images"][0]  # B, 3, W, W
		end_img = batch["images"][1]  # B, 3, W, W
		mean_img = batch["mean_video"][0]

		bs = start_img.shape[0]

		if isinstance(batch["index"], list):
			start_index = batch["index"][0]
			end_index = batch["index"][1]
		elif len(batch["index"].shape) > 1:
			start_index = batch["index"][:, 0]
			end_index = batch["index"][:, 1]
		else:
			start_index = batch["index"][0]
			end_index = batch["index"][1]

		if torch.cuda.is_available():
			start_img = start_img.cuda()
			end_img = end_img.cuda()
			mean_img = mean_img.cuda()

		bg_img_f = self.net_bg(start_img)  # 2,out_channel,W,W
		bg_img_f = nn.Tanh()(bg_img_f)

		bg_img_p = self.net_bg(end_img)  # 2,out_channel,W,W
		bg_img_p = nn.Tanh()(bg_img_p)


		# Regreesed Motion
		if "train_motion" in self.opt and self.opt.train_motion:
			flow_gt = batch["motions"]
			batch4motion = {
				"images": [start_img],
				"motions": batch["motions"],
			}
			motion_loss, outputs_motion = self.motion_regressor.forward(
				batch4motion)
			flow_uvm = outputs_motion["PredMotion"]
			if flow_uvm.shape[1] == 2:
				flow = flow_uvm.view(bs, 2, self.opt.W, self.opt.W)
				use_uvm = False
			elif flow_uvm.shape[1] == 3:
				flow = flow_uvm[:, :2, ...] * flow_uvm[:, 2:3, ...]
				flow = flow.view(bs, 2, self.opt.W, self.opt.W)
				use_uvm = True
			else:
				print("Not Implemented")
			flow_scale = [self.opt.W / 256, self.opt.W / 256]
			flow = flow * torch.FloatTensor(flow_scale).view(1, 2, 1, 1).to(flow.device)  # train in W256
			flow = flow.view(bs, 2, -1)
		else:
			if batch["motions"].shape[1] == 2:
				flow = batch["motions"].view(bs, 2, -1).cuda()
				use_uvm = False
			elif batch["motions"].shape[1] == 3:
				use_uvm = True
				flow_uvm = batch["motions"]
				flow = batch["motions"][:, :2, ...].clone()
				flow = flow * batch["motions"][:, 2:3, ...]
				flow = flow.view(bs, 2, -1).cuda()


		loss = {}
		loss["Total Loss"] = 0.0
		if self.opt.MVloss>0.0:
			loss_bgimg = self.loss_function(bg_img_f, mean_img)
			for i, loss_name in enumerate(loss_bgimg):
				if "Perceptual" in loss_name:
					loss[loss_name + "_bg"] = loss_bgimg[loss_name]
				elif "L1" in loss_name:
					loss[loss_name + "_bg"] = loss_bgimg[loss_name]
				elif "Total" in loss_name:
					loss[loss_name] += loss_bgimg[loss_name] * self.opt.MVloss #* 0.3

		motion_speed = (flow[:, 0:1, :] ** 2 + flow[:, 1:2, :] ** 2).sqrt().view(bs, 1, self.opt.W, self.opt.W)
		# motion_speed_norm = motion_speed / motion_speed.max()
		small_motion_alpha = (motion_speed < motion_speed.mean([1,2,3],True)*0.1).float()

		if self.opt.StaticRegionInputImageSupervision>0.0:
			loss["StaticRegionInputImageSupervision"] = (nn.L1Loss()(bg_img_f, start_img)* small_motion_alpha).mean()
			loss["Total Loss"] += loss["StaticRegionInputImageSupervision"] * self.opt.StaticRegionInputImageSupervision #* 0.3
		if self.opt.MovingRegionMVloss>0.0:
			loss["MovingRegionMVloss"] = (nn.L1Loss()(bg_img_f, mean_img)* small_motion_alpha).mean()
			loss["Total Loss"] += loss["MovingRegionMVloss"] * self.opt.MovingRegionMVloss #* 0.3
		pred_dict = {
			"PredImg": bg_img_f,
			"OutputImg": mean_img,
		}
		for key in pred_dict.keys():
			if isinstance(pred_dict[key], tuple):
				pred_dict[key] = pred_dict[key][0]

		return loss, pred_dict

	def forward_flow(self, batch,):
		if "images" in batch.keys():
			image = batch["images"][0].cuda()

		bg_img = self.net_bg(image)  # 2,out_channel,W,W
		bg_img = nn.Tanh()(bg_img)

		pred_dict = {"PredImg": bg_img}
		return pred_dict

