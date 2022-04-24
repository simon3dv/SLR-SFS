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
from PIL import Image, ImageDraw
import json
from utils.utils import VideoReader, load_compressed_tensor
PATH = os.getcwd()
sys.path.append(os.path.join(PATH))


def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(np.frombuffer(buffer=strFlow, dtype=np.float32, count=1, offset=0) == 202021.25)

    intWidth = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=4)[0]
    intHeight = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=8)[0]

    return np.frombuffer(buffer=strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])


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
    colorjitter = random.random() > 0.5
    if "use_color_jitter" in opt and opt.use_color_jitter:
        colorjitter = random.random() > 0.5
    else:
        colorjitter = False
    colorjitter_params={}
    colorjitter_params['brightness'] = (torch.rand(1) * 0.2 + 1.0).numpy()[0]
    colorjitter_params['contrast'] = (torch.rand(1) * 0.2 + 1.0).numpy()[0]
    colorjitter_params['saturation'] = (torch.rand(1) * 0.2 + 1.0).numpy()[0]
    colorjitter_params['hue'] = (torch.rand(1) * 0.05).numpy()[0]
        #brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05

    return {'crop_pos': (x, y), 'crop_size':crop_size, 'flip': flip, 'colorjitter':colorjitter, 'colorjitter_params':colorjitter_params}


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

def __colorjitter(img, colorjitter, colorjitter_params):
    if colorjitter:
        brightness = colorjitter_params['brightness']#0.2
        contrast = colorjitter_params['contrast']#0.2
        saturation = colorjitter_params['saturation']#0.2
        hue = colorjitter_params['hue']#0.05
        return transforms.ColorJitter(brightness=[brightness,brightness],
                                      contrast=[contrast,contrast],
                                      saturation=[saturation,saturation],
                                      hue=[hue,hue])(img)
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
    if "use_color_jitter" in opt and opt.use_color_jitter:
        transform_list.append(transforms.Lambda(lambda img: __colorjitter(img, params['colorjitter'], params['colorjitter_params'])))
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
        start_index = self.rng.randint(0,N//3)
        end_index = self.rng.randint(N//3*2,N)
        middle_index = self.rng.randint(start_index, end_index)
        flowpath = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_motion.pth")
        flow = load_compressed_tensor(flowpath)#torch ([1, 2, 720, 1280])
        gt_motion = flow.clone()
        height, width = flow.shape[2], flow.shape[3]
        #flow = flow.permute((2, 0, 1)).contiguous()#(2,W,W)

        # mask_rock
        labelpath = os.path.join(self.opt.rock_label_data_path, id+".png.json")
        mask_rock = np.zeros((flow.shape[2], flow.shape[3],1))
        if os.path.exists(labelpath):
            #print("read",labelpath)
            with open(labelpath, 'r') as load_f:
                label = json.load(load_f)
            width = label["width"]
            height = label["height"]
            label_results = label['step_1']['result']
            for i, label_result in enumerate(label_results):
                label_pointlist = label_result['pointList']
                polygon = []
                for point in label_pointlist:
                    polygon.append((point['x'], point['y']))
                label1mask = Image.new('L', (width, height), 0)
                ImageDraw.Draw(label1mask).polygon(polygon, outline=1, fill=1)
                label1mask = np.array(label1mask)  # (1080, 1920)
                label1mask = np.expand_dims(label1mask, 2)
                # label1mask = mask_transform(label1mask)
                # label1mask = label1mask.squeeze().cuda()
                """
                if i == 0:
                    mask_rock = label1mask
                else:
                    mask_rock += label1mask
                """
                mask_rock += label1mask
            mask_rock[mask_rock>1.0] = 1.0
            mask_rock = torch.FloatTensor(mask_rock).permute(2,0,1).unsqueeze(0)
        else:
            mask_rock = torch.zeros((flow.shape[0], 1, flow.shape[2], flow.shape[3]))


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


        # Hint
        if "use_online_hint" and self.opt.use_online_hint and (not self.isval):
            pass
        else:
            hintpath = os.path.join(self.opt.train_data_path[0], self.dataset, id+"_sparse_motion.flo")
            #hint = load_compressed_tensor(hintpath)#torch ([1, 2, 720, 1280])
            hint = torch.FloatTensor(read_flo(hintpath))  # (1024, 1920, 2)
            hint = hint.permute((2, 0, 1)).contiguous().unsqueeze(0)

        if self.isval:
            #Resize (1920,1024) to (512,512)
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
                    max_hint = int(1+self.rng.randint(10))
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

                    sigma = self.rng.randint(height // (max_hint*3), height // max_hint)
                    hint_y = hint_y.long()
                    hint_x = hint_x.long()
                    for i_hint in range(max_hint):
                        # sparse_motion_i = torch.zeros(gt_motion.shape).cuda()
                        # sparse_motion_i[:,:,hint_x[i_hint],hint_y[i_hint]] = gt_motion[:,:,hint_x[i_hint],hint_y[i_hint]]
                        # dense_motion_i = torch.zeros(gt_motion.shape).cuda().view(1,2,-1)
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
            #Crop (1920,1024) to (crop_size) and Resize to (512,512)
            if 'crop' in self.opt.resize_or_crop and 'resize' in self.opt.resize_or_crop:
                W = params['crop_size']
                crop_pos = params['crop_pos']
                hint_scale = [self.opt.W/W, self.opt.W/W]
                hint = hint[:, :, crop_pos[1]:crop_pos[1] + W, crop_pos[0]:crop_pos[0] + W]
            elif 'resize' in self.opt.resize_or_crop:
                hint_scale = [self.opt.W/flow.shape[3], self.opt.W/flow.shape[2]]

            #flow = torch.from_numpy(flow)
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


        if not self.isval:
            if 'crop' in self.opt.resize_or_crop and 'resize' in self.opt.resize_or_crop:
                W = params['crop_size']
                crop_pos = params['crop_pos']
                mask_rock = mask_rock[:, :, crop_pos[1]:crop_pos[1] + W, crop_pos[0]:crop_pos[0] + W]
            if params['flip']:
                mask_rock = torch.flip(mask_rock,[3])
        mask_rock = F.interpolate(mask_rock, (self.opt.W, self.opt.W), mode='nearest')
        mask_rock = mask_rock[0]


        for i in range(0, self.num_views):
            if i == 0:
                t_index = start_index
            elif i == 1:
                t_index = middle_index
            else:
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
                #"flows": flows,
                "motions":flow,
                "index": [start_index, middle_index,end_index],
                 "isval": self.isval,
                 "hints":hint,
                 "mask_rock": mask_rock,
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

