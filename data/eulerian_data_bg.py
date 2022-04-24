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
from utils.utils import VideoReader, load_compressed_tensor
PATH = os.getcwd()
sys.path.append(os.path.join(PATH))

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]


def get_params(opt, size=(1920,1024), crop_size=1024):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = 1024
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = 1024
        new_h = 1024 * h // w

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    if opt.no_flip:
        flip = False
    return {'crop_pos': (x, y), 'crop_size':crop_size, 'flip': flip}


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_transform(opt, size, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], params['crop_size'])))
    if 'resize' in opt.resize_or_crop:
        osize = [opt.W, opt.W]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.W, method)))
    if not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

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
