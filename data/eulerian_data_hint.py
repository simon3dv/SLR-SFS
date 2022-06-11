import av
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
import torchvision.transforms as transforms
import numpy as np
import pickle
import lz4framed
from PIL import Image
import os
import time
import sys
import random
import torch.nn.functional as F
from utils.utils import read_flo, load_compressed_tensor, VideoReader, get_params, get_transform
PATH = os.getcwd()
sys.path.append(os.path.join(PATH))

class Liquid(data.Dataset):
    """
    """

    def __init__(
        self, dataset, opts=None, num_views=3, seed=0, vectorize=False
    ):
        # Now go through the dataset
        # Raw Image Size
        self.opt=opts
        self.W = 1280
        self.H = 720
        if self.opt.use_single_scene:
            self.imageset = np.array(sorted([x[:11] for x in os.listdir(
                os.path.join(self.opt.train_data_path[0], dataset)
            ) if 'mp4' in x and x[:5]==self.opt.scene_id]))
        else:
            self.imageset = np.array(sorted([x[:11] for x in os.listdir(
                os.path.join(self.opt.train_data_path[0], dataset)
            ) if 'mp4' in x]))
        self.rng = np.random.RandomState(seed)
        self.base_file = opts.train_data_path[0]

        self.num_views = num_views

        self.input_transform = Compose(
            [
                Resize((opts.W, opts.W),Image.BICUBIC),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.dataset = "train"
        self.isval = False
        self.opt = opts

    def __len__(self):
        return max(2**15,len(self.imageset))

    def __getitem__(self, index):
        crop_size = 720
        params = get_params(self.opt, size=(self.W, self.H),
                            crop_size=crop_size)
        index = self.rng.randint(self.imageset.shape[0])
        id = self.imageset[index]

        rgbs = []
        flows = []
        video_file = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_gt.mp4")
        video_reader = VideoReader(video_file, None, "mp4")
        N = len(video_reader)
        start_index = self.rng.randint(0,N//3)
        end_index = self.rng.randint(N//3*2,N)
        middle_index = self.rng.randint(start_index, end_index)
        flowpath = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_motion.pth")
        flow = load_compressed_tensor(flowpath)
        gt_motion = flow.clone()
        height, width = flow.shape[2], flow.shape[3]
        if self.isval:
            flow_scale = [self.opt.W/flow.shape[3], self.opt.W/flow.shape[2]]
        else:
            if 'crop' in self.opt.resize_or_crop and 'resize' in self.opt.resize_or_crop:
                W = params['crop_size']
                crop_pos = params['crop_pos']
                flow_scale = [self.opt.W/W, self.opt.W/W]
                flow = flow[:, :, crop_pos[1]:crop_pos[1] + W, crop_pos[0]:crop_pos[0] + W]
            elif 'resize' in self.opt.resize_or_crop:
                flow_scale = [self.opt.W/flow.shape[3], self.opt.W/flow.shape[2]]

            if params['flip']:
                flow = torch.flip(flow,[3])
                flow[:, 0, :, :] *= -1
        flow = flow * torch.FloatTensor(flow_scale).view(1,2,1,1)
        flow = F.interpolate(flow, (self.opt.W, self.opt.W), mode='bilinear')
        flow = flow.view(2, -1)  # (2, WW)


        # Hint
        if "use_online_hint" and self.opt.use_online_hint and (not self.isval):
            pass
        else:
            hintpath = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_sparse_motion.flo")
            hint = torch.FloatTensor(read_flo(hintpath))
            hint = hint.permute((2, 0, 1)).contiguous().unsqueeze(0)

        if self.isval:
            hint_scale = [self.opt.W/width, self.opt.W/height]
        else:
            if "use_online_hint" and self.opt.use_online_hint:
                speed = 1
                bs = 1
                xs = torch.linspace(0, width - 1, width)
                ys = torch.linspace(0, height - 1, height)
                xs = xs.view(1, 1, width).repeat(1, height, 1)
                ys = ys.view(1, height, 1).repeat(1, 1, width)
                xys = torch.cat((xs, ys), 1).view(2, -1)  # (2,WW)
                gt_motion_speed = (gt_motion[:, 0:1, ...] ** 2 + gt_motion[:, 1:2, ...] ** 2).sqrt().view(bs, 1, height,
                                                                                                          width)
                # big_motion_alpha = (gt_motion_speed > gt_motion_speed.mean([1, 2, 3], True) * 0.1 -1e-8).float() 0.2161635
                big_motion_alpha = (gt_motion_speed > 0.2161635).float()
                if int(big_motion_alpha.sum().long()) < 10:
                    dense_motion = torch.zeros(gt_motion.shape)
                else:
                    max_hint = int(1+self.rng.randint(5))
                    estimator = KMeans(n_clusters=max_hint)
                    # index = torch.randint(0, int(big_motion_alpha.sum()), (max_hint,))
                    hint_y = torch.zeros((max_hint,))
                    hint_x = torch.zeros((max_hint,))
                    big_motion_xys = xys[:, torch.where(big_motion_alpha.view(1, 1, height * width))[2]]  # 2, M
                    X = big_motion_xys.permute(1, 0).cpu().detach().numpy()
                    estimator.fit(X)
                    labels = estimator.labels_
                    for i in range(max_hint):
                        selected_xy = X[labels == i].mean(0)
                        hint_y[i] = int(selected_xy[1])
                        hint_x[i] = int(selected_xy[0])

                    dense_motion = torch.zeros(gt_motion.shape).view(1, 2, -1)
                    dense_motion_norm = torch.zeros(gt_motion.shape).view(1, 2, -1)

                    sigma = self.rng.randint(height // (max_hint*2), height // (max_hint/2))
                    hint_y = hint_y.long()
                    hint_x = hint_x.long()
                    for i_hint in range(max_hint):
                        dist = ((xys - xys.view(2, height, width)[:, hint_y[i_hint], hint_x[i_hint]].unsqueeze(
                            1)) ** 2).sum(0, True).sqrt()  # 1,W*W
                        weight = (-(dist / sigma) ** 2).exp().unsqueeze(0)
                        dense_motion += weight * gt_motion[:, :, hint_y[i_hint], hint_x[i_hint], ].unsqueeze(2)
                        dense_motion_norm += weight
                        # torchvision.utils.save_image(weight.view(1, 1, 256, 256), "dist_Weight.png")
                    dense_motion_norm[dense_motion_norm == 0.0] = 1.0  # = torch.clamp(dense_motion_norm,min=1e-8)
                    dense_motion = dense_motion / dense_motion_norm
                    dense_motion = dense_motion.view(1, 2, height, width) * big_motion_alpha
                hint = dense_motion
            if 'crop' in self.opt.resize_or_crop and 'resize' in self.opt.resize_or_crop:
                W = params['crop_size']
                crop_pos = params['crop_pos']
                hint_scale = [self.opt.W/W, self.opt.W/W]
                hint = hint[:, :, crop_pos[1]:crop_pos[1] + W, crop_pos[0]:crop_pos[0] + W]
            elif 'resize' in self.opt.resize_or_crop:
                hint_scale = [self.opt.W/flow.shape[3], self.opt.W/flow.shape[2]]

            if params['flip']:
                hint = torch.flip(hint,[3])
                hint[:, 0, :, :] *= -1

        hint = hint * torch.FloatTensor(hint_scale).view(1,2,1,1)
        hint = F.interpolate(hint, (self.opt.W, self.opt.W), mode='bilinear',align_corners=False)
        hint = hint.view(2, -1)  # (2, WW)

        if self.opt.use_gt_mean_img or self.opt.MVloss > 0.0:
            mean_video_path = os.path.join(self.opt.train_data_path[0],"avr_image", id+".png")
            mean_video = Image.open(mean_video_path)
            if self.isval:
                mean_video = self.input_transform(mean_video)
            else:
                transforms4train = get_transform(self.opt,
                                                 (self.W, self.H),
                                                 params)
                mean_video = transforms4train(mean_video)
        for i in range(0, self.num_views):
            if i == 0:
                t_index = start_index
            elif i == 1:
                t_index = middle_index
            else:
                t_index = end_index
            image = torch.FloatTensor(video_reader[t_index][0]).permute(2,0,1)/255.0
            image = transforms.ToPILImage()(image)
            if self.isval:
                image = self.input_transform(image)
            else:
                transforms4train = get_transform(self.opt,
                                                 (self.W,self.H),
                                                 params)
                image = transforms4train(image)

            rgbs += [image]
        batch = {"images": rgbs,
                "motions":flow,
                "index": [start_index, middle_index,end_index],
                 "isval": self.isval,
                 "hints":hint,
        }
        if self.opt.use_gt_mean_img or self.opt.MVloss > 0.0:
            batch["mean_video"] = [mean_video]
        return batch

    def totrain(self, epoch):
        self.isval = False
        self.dataset = "train"

        if self.opt.use_single_scene:
            self.imageset = np.array(sorted([x[:11] for x in os.listdir(
                os.path.join(self.opt.train_data_path[0], self.dataset)
            ) if 'mp4' in x and x[:5]==self.opt.scene_id]))
        else:
            self.imageset = np.array(sorted([x[:11] for x in os.listdir(
                os.path.join(self.opt.train_data_path[0], self.dataset)
            ) if 'mp4' in x]))
        self.rng = np.random.RandomState(epoch)

    def toval(self, epoch):
        self.isval = True
        if self.opt.use_single_scene:
            self.dataset = "train"
            self.imageset = np.array(sorted([x[:11] for x in os.listdir(
                os.path.join(self.opt.train_data_path[0], self.dataset)
            ) if 'mp4' in x and x[:5]==self.opt.scene_id ]))
        else:
            self.dataset = "validation"
            self.imageset = np.array(sorted([x[:11] for x in os.listdir(
                os.path.join(self.opt.train_data_path[0], self.dataset)
            ) if 'mp4' in x]))

        self.rng = np.random.RandomState(epoch)


