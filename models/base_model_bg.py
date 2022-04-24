# Code adapted from:
# https://github.com/facebookresearch/synsin
import torch
import torch.nn as nn
from torch.nn import init
from models.losses.gan_loss import DiscriminatorLoss
import os
class BaseModelBG(nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model

        self.opt = opt

        if opt.discriminator_losses != '0' :
            self.use_discriminator = True

            self.netD_bg = DiscriminatorLoss(opt)

            if opt.isTrain:
                self.optimizer_D = torch.optim.Adam(
                    list(self.netD_bg.parameters()),
                    lr=opt.lr_d,
                    betas=(opt.beta1, opt.beta2),
                )
                self.optimizer_G = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=opt.lr_g,
                    betas=(opt.beta1, opt.beta2),
                )
        else:
            self.use_discriminator = False
            self.optimizer_G = torch.optim.Adam(
                list(self.model.parameters()),
                lr=opt.lr_g,
                betas=(0.99, opt.beta2),
            )

        if opt.isTrain:
            self.old_lr = opt.lr

        if opt.init:
            self.init_weights()

    def init_weights(self, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if self.opt.init == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif self.opt.init == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif self.opt.init == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif self.opt.init == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self.opt.init == "orthogonal":
                    init.orthogonal_(m.weight.data)
                elif self.opt.init == "":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % self.opt.init
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.opt.init, gain)

    def update_learning_rate(self):
        #G
        lr_g_decay = self.lr_g / self.opt.niter_decay
        lr_g = self.lr_g - lr_g_decay
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_g
        self.lr_g = lr_g
        if self.opt.discriminator_losses != '0':
            #D
            lr_d_decay = self.lr_d / self.opt.niter_decay
            lr_d = self.lr_d - lr_d_decay
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr_d
            self.lr_d = lr_d

    def __call__(
        self, dataloader, isval=False, num_steps=1, return_batch=False
    ):
        weight = 1.0 / float(num_steps)
        if isval:
            #tic = time.time()
            batch = next(dataloader)
            #tic_loader = time.time() - tic
            #tic = time.time()
            t_losses, output_images = self.model(batch)
            #tic_forward = time.time() - tic
            # dict_keys(['ssim', 'psnr', 'Perceptual', 'Total Loss', 'L1'])
            # dict_keys(['InputImg', 'OutputImg', 'PredImg', 'PredDepth'])
            if self.opt.normalize_image:
                for k in output_images.keys():
                    if "Img" in k:# [-1,1] to [0,1]
                        output_images[k] = 0.5 * output_images[k] + 0.5
            if return_batch:
                return t_losses, output_images, batch
            return t_losses, output_images

        self.optimizer_G.zero_grad()

        if self.use_discriminator:
            all_output_images = []
            for j in range(0, num_steps):
                batch = next(dataloader)
                t_losses, output_images = self.model(batch)

                g_losses = self.netD_bg.run_generator_one_step(
                    output_images["PredImg"], output_images["OutputImg"]
                )

                (
                    g_losses["Total Loss"] / weight
                    + t_losses["Total Loss"] / weight
                ).mean().backward()

                all_output_images += [output_images]
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            for step in range(0, num_steps):
                d_losses = self.netD_bg.run_discriminator_one_step(
                    all_output_images[step]["PredImg"],
                    all_output_images[step]["OutputImg"],
                )
                (d_losses["Total Loss"] / weight).mean().backward()
            # Apply orthogonal regularization from BigGan
            self.optimizer_D.step()

            g_losses.pop("Total Loss")
            d_losses.pop("Total Loss")
            t_losses.update(g_losses)
            t_losses.update(d_losses)
        else:
            for step in range(0, num_steps):
                t_losses, output_images = self.model(next(dataloader))
                (t_losses["Total Loss"] / weight).mean().backward()
            self.optimizer_G.step()

        if self.opt.normalize_image:
            for k in output_images.keys():
                if "Img" in k:
                    output_images[k] = 0.5 * output_images[k] + 0.5

        return t_losses, output_images
