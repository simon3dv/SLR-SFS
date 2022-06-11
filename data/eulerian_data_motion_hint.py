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
from sklearn.cluster import KMeans
import torchvision
PATH = os.getcwd()
sys.path.append(os.path.join(PATH))
from utils.utils import read_flo, load_compressed_tensor, get_params, get_transform
from torchvision.transforms import InterpolationMode
from utils.flow_utils import flow2img, writeFlow

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
                Resize((opts.motionH, opts.motionW),InterpolationMode.BICUBIC),
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
        index = self.rng.randint(self.imageset.shape[0])
        id = self.imageset[index]

        rgbs = []
        flows = []
        img_path = os.path.join(self.opt.train_data_path[0], self.dataset, id + "_input.jpg")
        image = Image.open(img_path)
        width, height = image.size
        crop_size = height
        params = get_params(self.opt, size=(width, height),
                            crop_size=crop_size)
        if self.isval:
            image = self.input_transform(image)
        else:
            transforms4train = get_transform(self.opt,
                                             (width, height),
                                             params)
            image = transforms4train(image)
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
        flow = F.interpolate(flow, (self.opt.W, self.opt.W), mode='bilinear',align_corners=False)
        flow = flow.view(2, -1)  # (2, WW)

        # Hint
        if "use_online_hint" and self.opt.use_online_hint and (not self.isval):
            pass
        else:
            hintpath = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_sparse_motion.flo")
            hint = torch.FloatTensor(read_flo(hintpath))  # (1024, 1920, 2)
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
                    dense_motion_norm[dense_motion_norm == 0.0] = 1.0
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


        rgbs += [image]
        batch = {"images": rgbs,
                "motions":flow,
                "hints":hint,
                 "isval": self.isval,
        }
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



if __name__=='__main__':
    import sys
    import os
    rootdir = ""
    name = "00980_00000"
    names = sorted([x for x in os.listdir(rootdir) if "_motion.pth" in x and "sparse_motion" not in x
                     ])
    
    print(names)
    np.random.seed(5)

    for i_name, name in enumerate(names):
        flow_file = os.path.join(rootdir, name)
        print("Processing %s."%name)
        out_file =  os.path.join(rootdir, name[:-11]+"_sparse_motion.flo")
        flow = load_compressed_tensor(flow_file)
        print("Mean flow : %.2f" % flow.abs().mean())
        speed = 1
        bs = 1
        height, width = flow.shape[2],flow.shape[3]

        xs = torch.linspace(0, width - 1, width).cuda()
        ys = torch.linspace(0, height - 1, height).cuda()
        xs = xs.view(1, 1, width).repeat(1, height, 1)
        ys = ys.view(1, height, 1).repeat(1, 1, width)
        xys = torch.cat((xs, ys), 1).view(2, -1)  # (2,WW)

        gt_motion = flow.cuda().view(1,2,height,width)
        gt_motion_speed = (gt_motion[:, 0:1, ...] ** 2 + gt_motion[:, 1:2, ...] ** 2).sqrt().view(bs, 1, height, width)
        big_motion_alpha = (gt_motion_speed > 0.2161635).float()
        if int(big_motion_alpha.sum().long()) < 5:
            dense_motion = torch.zeros(gt_motion.shape).cuda().view(1, 2, -1)
            continue
        else:
            max_hint = 5
            estimator = KMeans(n_clusters=max_hint)
            #index = torch.randint(0, int(big_motion_alpha.sum()), (max_hint,))
            hint_y = torch.zeros((max_hint,))
            hint_x = torch.zeros((max_hint,))
            big_motion_xys = xys[:,torch.where(big_motion_alpha.view(1,1,height*width))[2]] # 2, M
            X = big_motion_xys.permute(1,0).cpu().detach().numpy()
            estimator.fit(X)
            labels = estimator.labels_
            for i in range(max_hint):
                selected_xy = X[labels==i].mean(0)
                hint_y[i] = int(selected_xy[1])
                hint_x[i] = int(selected_xy[0])


            dense_motion = torch.zeros(gt_motion.shape).cuda().view(1, 2, -1)
            dense_motion_norm = torch.zeros(gt_motion.shape).cuda().view(1, 2, -1)

            sigma = height//5
            hint_y = hint_y.long()
            hint_x = hint_x.long()
            for i_hint in range(max_hint):
                dist = ((xys - xys.view(2, height, width)[:, hint_y[i_hint], hint_x[i_hint]].unsqueeze(
                    1)) ** 2).sum(0, True).sqrt()  # 1,W*W
                weight = (-(dist / sigma) ** 2).exp().unsqueeze(0)
                dense_motion += weight * gt_motion[:, :, hint_y[i_hint], hint_x[i_hint], ].unsqueeze(2)
                dense_motion_norm += weight
                #torchvision.utils.save_image(weight.view(1, 1, 256, 256), "dist_Weight.png")
            dense_motion_norm[dense_motion_norm == 0.0] = 1.0  # = torch.clamp(dense_motion_norm,min=1e-8)
            dense_motion = dense_motion / dense_motion_norm
            dense_motion = dense_motion.view(1, 2, height, width) * big_motion_alpha
        writeFlow(out_file, dense_motion[0].permute(1, 2, 0).cpu().detach().numpy())

        img8 = flow2img(gt_motion[0].permute(1, 2, 0).cpu().detach().numpy())
        img8 = torch.FloatTensor(img8).permute(2, 0, 1).unsqueeze(0)
        torchvision.utils.save_image(img8 / 255, os.path.join(out_file.replace("_sparse_motion.flo","_motion.png")))


        img8 = flow2img(dense_motion[0].permute(1, 2, 0).cpu().detach().numpy())
        img8 = torch.FloatTensor(img8).permute(2, 0, 1).unsqueeze(0)
        torchvision.utils.save_image(img8 / 255, os.path.join(out_file.replace("_sparse_motion.flo","_sparse_motion.png")))
