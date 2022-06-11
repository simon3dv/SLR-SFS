# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch.nn as nn
import copy

from models.networks.architectures import (
    ResNetBGDecoder,
    ResNetDecoderPconv,
    ResNetDecoderPconv2,
    ResNetDecoder,
    ResNetDecoderAlpha,
    ResNetEncoder,
    ResNetEncoder_with_Z,
)

EPS = 1e-2

def get_encoder(opt, downsample=True):
    if "unet_encoder" in opt.refine_model_type:
        encoder = UNetEncoder16(opt, channels_in=3, channels_out=opt.out_channel)
    else:
        if "skip_warp" in opt.refine_model_type:
            pass
        else:
            encoder = ResNetEncoder_with_Z(
                opt, channels_in=3, channels_out=opt.out_channel, downsample=downsample
            )
    return encoder


class Identity(nn.Module):
    def forward(self, input):
        return input


def get_decoder(opt):
    if opt.use_rgb_features:
        channel_in = 3
    else:
        channel_in = opt.ngf
    if "addtional_decoder_input" in opt:
        channel_in += opt.addtional_decoder_input
    channels_out = 3
    if "addtional_decoder_output" in opt:
        channels_out += opt.addtional_decoder_output

    if "unet_decoder" in opt.refine_model_type:
        if "pconv" in opt.refine_model_type:
            if "skip_warp" in opt.refine_model_type:
                pass
            if "unet16" in opt.refine_model_type:
                decoder = UNetDecoder16Pconv2(opt, channels_in=channel_in, channels_out=3)
        else:
            if "unet16" in opt.refine_model_type:
                decoder = UNetDecoder16(opt, channels_in=channel_in, channels_out=3)

    elif "resnet" in opt.refine_model_type:
        if "pconv" in opt.refine_model_type:
            decoder = None
            if "pconv2" in opt.refine_model_type and "skip_warp" not in opt.refine_model_type :
                print("RESNET decoder + pconv_v2.")
                decoder = ResNetDecoderPconv2(opt, channels_in=64, channels_out=3, use_tanh=False)#"nonorm" not in opt.refine_model_type)

            if "skip_warp" in opt.refine_model_type:
                pass
        else:
            print("RESNET decoder.")
            decoder = ResNetDecoder(opt, channels_in=64, channels_out=3, use_tanh=False)#"nonorm" not in opt.refine_model_type)

    else:
        decoder = Identity()

    return decoder


def get_decoder_bg(opt):
    if opt.use_rgb_features:
        channel_in = 3
    else:
        channel_in = opt.ngf
    if "addtional_decoder_input" in opt:
        channel_in += opt.addtional_decoder_input
    channels_out = 3
    if "addtional_decoder_output" in opt:
        channels_out += opt.addtional_decoder_output
    if "unet" in opt.refine_model_type:
        pass
    elif "resnet" in opt.refine_model_type:
        if "pconv2" in opt.refine_model_type:
            decoder = ResNetDecoderPconv2(opt, channels_in=64, channels_out=3, use_tanh=False)
        else:
            decoder = ResNetDecoder(opt, channels_in=64, channels_out=3, use_tanh=False)
    else:
        decoder = Identity()

    return decoder

def get_net_bg(opt):
    decoder = ResNetBGDecoder(opt, channels_in=3, channels_out=3, use_tanh=False)

    return decoder



def get_alpha_encoder(opt):
    opt_alpha = copy.deepcopy(opt)
    opt_alpha.refine_model_type = opt.alpha_refine_model_type

    if opt.AKLloss >0.0:
        opt_alpha.out_channel = 3
    else:
        opt_alpha.out_channel = 2
    channels_in = 3
    encoder = ResNetEncoder(opt_alpha,channels_in=channels_in)

    return encoder

def get_alpha_decoder(opt):
    opt_alpha = copy.deepcopy(opt)
    opt_alpha.refine_model_type = opt.alpha_refine_model_type
    if opt.AKLloss >0.0:
        opt_alpha.out_channel = 3
    else:
        opt_alpha.out_channel = 2
    opt_alpha.addtional_decoder_input = 1
    if "decouple" in opt_alpha.refine_model_type:
        opt_alpha.addtional_decoder_input -= opt.ngf
    elif "image" in opt_alpha.refine_model_type:
        opt_alpha.addtional_decoder_input -= opt.ngf - 3
    opt_alpha.addtional_decoder_output = -2
    decoder = ResNetDecoderPconv2(opt_alpha, use_tanh=False)

    return decoder




def get_motion_SPADE(opt):
    opt_motion = copy.deepcopy(opt)
    opt_motion.refine_model_type = opt.motion_refine_model_type
    channels_in = 3
    if "use_mask_as_motion_input" in opt_motion and opt_motion.use_mask_as_motion_input:
        channels_in += 1
    if "use_hint_as_motion_input" in opt_motion and opt_motion.use_hint_as_motion_input:
        channels_in += 2
    decoder = ResNetEncoder(opt_motion,channels_in=channels_in)

    return decoder