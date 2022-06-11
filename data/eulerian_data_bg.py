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
from utils.utils import VideoReader, load_compressed_tensor, get_params
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
        #train:from 00001_00000_gt.mp4 to  00979_00120_gt.mp4
        #validation: from 00980_00000_gt.mp4 to 01010_00096_gt.mp4
        #format
        # jpg:
        # mp4:
        # pth:
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
        crop_size = 720 #random.randint((self.opt.W + self.H2) / 2, self.H2)
        params = get_params(self.opt, size=(self.W, self.H),
                            crop_size=crop_size)
        index = self.rng.randint(self.imageset.shape[0])
        id = self.imageset[index]
        # index = index % self.imageset.shape[0]

        rgbs = []
        flows = []
        video_file = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_gt.mp4")
        video_reader = VideoReader(video_file, None, "mp4")
        #middle_index = self.rng.randint(1,len(video_reader)-1) #[1,N-1)  i i
        #start_index = 0
        #end_index = 59 # no end_index
        N = len(video_reader)
        start_index = int(self.rng.randint(0, N-2,size=2).min())
        end_index = int(self.rng.randint(start_index+1, N,size=2).max())
        flowpath = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_motion.pth")
        flow = load_compressed_tensor(flowpath)#torch ([1, 2, 720, 1280])
        #flow = flow.permute((2, 0, 1)).contiguous()#(2,W,W)
        if self.isval:
            #Resize (1920,1024) to (512,512)
            flow_scale = [self.opt.W/flow.shape[3], self.opt.W/flow.shape[2]]
            #flow = torch.from_numpy(flow)
            #minn = min(flow.shape[0],flow.shape[1])
            #cropper = StaticCenterCrop(((flow.shape[0],flow.shape[1])), (minn,minn))
            #flow = cropper(flow)
        else:
            #Crop (1920,1024) to (crop_size) and Resize to (512,512)
            if 'crop' in self.opt.resize_or_crop and 'resize' in self.opt.resize_or_crop:
                W = params['crop_size']
                crop_pos = params['crop_pos']
                flow_scale = [self.opt.W/W, self.opt.W/W]
                flow = flow[:, :, crop_pos[1]:crop_pos[1] + W, crop_pos[0]:crop_pos[0] + W]
            elif 'resize' in self.opt.resize_or_crop:
                flow_scale = [self.opt.W/flow.shape[3], self.opt.W/flow.shape[2]]

            #flow = torch.from_numpy(flow)
            if params['flip']:
                flow = torch.flip(flow,[3])
                flow[:, 0, :, :] *= -1
        flow = flow * torch.FloatTensor(flow_scale).view(1,2,1,1)
        flow = F.interpolate(flow, (self.opt.W, self.opt.W), mode='bilinear')
        flow = flow.view(2, -1)  # (2, WW)

        if True:
            mean_video_path = os.path.join(self.opt.train_data_path[0],"avr_image", id+".png")
            mean_video = Image.open(mean_video_path)
            if self.isval:
                mean_video = self.input_transform(mean_video)
            else:
                transforms4train = get_transform(self.opt,
                                                 (self.W, self.H),
                                                 params)
                mean_video = transforms4train(mean_video)
        for i in range(0, 2):
            if i == 0:
                t_index = start_index
            elif i == 1:
                t_index = end_index
            image = torch.FloatTensor(video_reader[t_index][0]).permute(2,0,1)/255.0#torch.Size([3, 720, 1280])
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
                "index": [start_index,end_index],
                 "isval": self.isval,
        }
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
