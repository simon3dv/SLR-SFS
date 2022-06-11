import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(os.path.dirname(BASE_DIR))#,'../')
sys.path.append(ROOT_DIR)
import matplotlib.pyplot as plt
import quaternion
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel
#from models.base_model_joint_gan import BaseModel
from sklearn.cluster import KMeans
from options.options import get_model

from PIL import Image
import ipdb
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from models.projection.euler_integration_manipulator import EulerIntegration

import time
import pickle
import lz4framed
import json
from utils.flow_utils import flow2img, writeFlow
from utils.utils import read_flo, load_compressed_tensor

os.environ['DEBUG'] = '0'
torch.backends.cudnn.enabled = True

DEBUG_TIME = False
MODEL_NAMES = [
    'AnimatingModel',
]
bs = 1
class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

if __name__=='__main__':
    rootdir = "/home/SENSETIME/fansiming/Documents/AnimatingPicture"
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(rootdir,
                                  'data/VideosLiquid/Mini/DemoWaterfall/img/videoblocks-beautiful-scenic-landscape-of-the-gullfoss-waterfall-in-iceland-splashing-water-falling-down-from-the-cliff-with-foam_r9mnahtcez_1080__D/000031.jpg')
    if len(sys.argv) > 2:
        flow_file = sys.argv[2]
    else:
        flow_file = os.path.join(rootdir,
                                 '/home/SENSETIME/fansiming/Documents/AnimatingPicture/data/VideosLiquid/Mini/DemoWaterfall/flow_average/flow/000031.flo')
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    else:
        save_dir = os.path.join(rootdir, 'data/VideosLiquid/Mini/DemoWaterfall/warp')

    if len(sys.argv) > 4:
        warp_pretrained_model = sys.argv[4]
    else:
        warp_pretrained_model = '/mnt/lustre/fansiming/project/mysynsin/logging/animating/VideosLiquid/2021-03/animating_synsin_20210322_Z_channel_bs6_relu/models/lr0.00010_bs6_modelsoftmax_splating_splxyblending/noise_bnsync:spectral_batch_refresnet_256W8UpDown64_dunet_camxysFalse|False/_init_databoth_seed0/_multiFalse_losses1.0|l110.0|content_izFalse_alphaFalse__vol_ganpix2pixHD/model_epoch.pthbest100'

    if len(sys.argv) > 5:
        name = sys.argv[5]
    else:
        name = "DemoWaterfall"

    if len(sys.argv) > 6:
        W = int(sys.argv[6])
    else:
        W = 768

    if len(sys.argv) > 7:
        N = int(sys.argv[7])
    else:
        N = 120

    if len(sys.argv) > 8:
        speed = float(sys.argv[8])
    else:
        speed = 0.25

    if len(sys.argv) > 9:
        align_data = sys.argv[9]
    else:
        align_data = "None"#"/home/SENSETIME/fansiming/Documents/2020_H2/mysynsin/data/eulerian_data/gtmotion_warp_outputs/1221_unet_W256_align_train_001hole/align_max_frame.json"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    FRAME = N
    num_frames = FRAME
    MODEL_NAME = MODEL_NAMES[0]
    MODEL_PATH = warp_pretrained_model
    opts = torch.load(MODEL_PATH)['opts']
    opts.render_ids = [1]
    opts.W = W
    opts.bn_noise_misc = True
    model = get_model(opts)
    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    device = 'cuda:' + str(torch_devices[0])

    if 'sync' in opts.norm_G:
        model = convert_model(model)
        model = nn.DataParallel(model, torch_devices[0:1]).cuda()
    else:
        model = nn.DataParallel(model, torch_devices[0:1]).cuda()

    #  Load the original model to be tested
    model_to_test = BaseModel(model, opts)


    #model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    #  Load the original model to be tested
    restore_ckpt = {}
    model_dict = model_to_test.state_dict()
    ckpt_dict = torch.load(MODEL_PATH)['state_dict']
    for (k, v) in ckpt_dict.items():
        if ('xyzs' not in k) and ("Z_predictor") not in k:
            restore_ckpt[k] = v
            model_dict.update(restore_ckpt)
    model_to_test.load_state_dict(model_dict)
    model_to_test.eval()

    print("Loaded model")
    model_to_test.eval()

    print(opts)
    # I_0
    image_name = image_file[image_file.rfind("/") + 1:image_file.rfind(".")]
    #img = cv2.imread(image_file)
    img = Image.open(image_file)
    output_W, output_H = img.size
    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((opts.W, opts.W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)

    flows_f = []
    flows_p = []

    #flow = load_compressed_tensor(flow_file)  # (1024, 1920, 2)
    img = img.unsqueeze(0).cuda()
    if "use_mask_as_motion_input" in opts and opts.use_mask_as_motion_input:
        if ".pth" in flow_file:
            gt_motion = load_compressed_tensor(flow_file)
        elif ".flo" in flow_file:
            gt_motion = torch.FloatTensor(read_flo(flow_file))
            gt_motion = gt_motion.permute((2, 0, 1)).contiguous().unsqueeze(0)

        height, width = gt_motion.shape[2], gt_motion.shape[3]
        if "use_hint_as_motion_input" in opts and opts.use_hint_as_motion_input:
            speed = 1
            bs = 1
            xs = torch.linspace(0, width - 1, width)
            ys = torch.linspace(0, height - 1, height)
            xs = xs.view(1, 1, width).repeat(1, height, 1)
            ys = ys.view(1, height, 1).repeat(1, 1, width)
            xys = torch.cat((xs, ys), 1).view(2, -1)  # (2,WW)
            gt_motion_speed = (gt_motion[:, 0:1, ...] ** 2 + gt_motion[:, 1:2, ...] ** 2).sqrt().view(bs, 1, height,
                                                                                               width)
            #big_motion_alpha = (gt_motion_speed > gt_motion_speed.mean([1, 2, 3], True) * 0.1 -1e-8).float()#0.2161635
            big_motion_alpha = (gt_motion_speed > 0.2161635).float()
            #moving_region_mask = big_motion_alpha
            if int(big_motion_alpha.sum().long()) < 5:
                dense_motion = torch.zeros(gt_motion.shape)
            else:
                max_hint = 5
                np.random.seed(5)
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

                sigma = height / max_hint #/ 2
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
            big_motion_alpha =  F.interpolate(big_motion_alpha, (opts.W, opts.W), mode='nearest')
            hint = F.interpolate(hint, (opts.W, opts.W))
            flow = model_to_test.model.module.motion_regressor.forward_flow(img, big_motion_alpha.cuda(), hint.cuda())["PredMotion"]
        else:
            gt_motion = F.interpolate(gt_motion, (opts.W, opts.W))
            motion_speed = (gt_motion[:, 0:1, :, :] ** 2 + gt_motion[:, 1:2, :, :] ** 2).sqrt().view(bs, 1, opts.W,                                            opts.W)
            small_motion_alpha = (motion_speed < motion_speed.mean([1, 2, 3], True) * 0.1).float()
            big_motion_alpha = 1.0 - small_motion_alpha
            flow = model_to_test.model.module.motion_regressor.forward_flow(img,big_motion_alpha)["PredMotion"]

    else:
        flow = model_to_test.model.module.motion_regressor.forward_flow(img)["PredMotion"]
    print("Mean flow : %.2f"%flow.abs().mean())
    flow_scale = [opts.W / flow.shape[3] * speed, opts.W / flow.shape[2] * speed]
    flow = flow * torch.FloatTensor(flow_scale).view(1,2,1,1).cuda()
    minn = min(flow.shape[3], flow.shape[2])
    flow = flow.view(1, 2, -1)  # (2, WW)


    if DEBUG_TIME:
        from utils.utils import AverageMeter
        time_dict = {
            't_forward_batch': AverageMeter(),
            't_load_batch': AverageMeter(),
            't_encoder': AverageMeter(),
            't_depth_regressor': AverageMeter(),
            't_motion_regressor': AverageMeter(),
            't_softmax_splating': AverageMeter(),
            't_euler_integration': AverageMeter(),
            't_decoder': AverageMeter()
        }
        torch.cuda.synchronize()
        tic = time.time()
    else:
        time_dict = False
    flow = flow.cuda()
    euler_integration = EulerIntegration(opts)
    model_to_test.model.module.eval()
    with torch.no_grad():
        # encoder
        if DEBUG_TIME:
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            tic = time.time()
        start_fs = model_to_test.model.module.encoder(img)
        #end_fs = start_fs

        if DEBUG_TIME:
            torch.cuda.synchronize()
            time_dict['t_encoder'].update(time.time() - tic)
            torch.cuda.synchronize()
            tic = time.time()

        if align_data != "None":
            with open(align_data, "r") as f:
                align_dict = json.load(f)
            frame = align_dict[name]
            flow = flow * frame / FRAME

        # Euler Integration
        #flow_f = euler_integration.forward_all(flow.view(2,-1), FRAME)
        #flow_p = euler_integration.forward_all(-flow.view(2,-1), FRAME+1)[1:]

        if DEBUG_TIME:
            torch.cuda.synchronize()
            time_dict['t_euler_integration'].update(time.time() - tic)
            torch.cuda.synchronize()
            tic = time.time()
        batch = {"features": [start_fs],
                 "images":[img],
                 'motions': [flow.view(1,2,opts.W,opts.W)],
                 #"depths": [torch.ones(img[:, 0:1, :, :].shape)+ torch.rand(img[:, 0:1, :, :].shape) * 1e-4],
                 'index': None}

        """
        #Write flow
        save_flow = os.path.join(save_dir,"Motion.flo")
        flow_np = flow[0].view(2,opts.W,opts.W).permute([1,2,0]).detach().cpu().numpy()
        writeFlow(save_flow, flow_np)
        flowimg = flow2img(flow_np)
        save_img = os.path.join(save_dir,"Motion-vis.png")
        plt.imsave(save_img, flowimg)
        """

        for t in tqdm(range(0, FRAME//bs)):
            batch['index'] = torch.Tensor([0, t*bs, FRAME - 1]).long().view(1,3)
            for tplus in range(1, bs):
                batch['index'] = torch.cat((batch['index'],torch.Tensor([0, t*bs+tplus, FRAME - 1]).long().view(1,3)), 0)

            with torch.no_grad():
                pred_imgs = model_to_test.model.module.forward_flow(batch)
            if "use_mask_as_motion_input" in opts and opts.use_mask_as_motion_input:
                pred_imgs['MovingRegionMask'] = big_motion_alpha
                pred_imgs['HintMotion'] = hint
            pred_imgs['Motion'] = flow.view(1,2,opts.W,opts.W)
            for key in pred_imgs.keys():
                if "Img" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W),mode='bilinear')
                    PredImg = (PredImg.permute(0,2,3,1).cpu().numpy() * 0.5 + 0.5) * 255 # bs, 3, 672, 378
                elif "Alpha" in key:#or "Z_" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W), mode='bilinear')
                    PredImg = (PredImg.permute(0,2,3,1).cpu().numpy()) * 255
                elif "Z_" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W),mode='bilinear')
                    PredImg = (PredImg - PredImg.min()) / (PredImg.max()-PredImg.min())
                    PredImg = (PredImg.permute(0, 2, 3, 1).cpu().numpy()) * 255
                elif "Motion" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W),mode='bilinear')
                    PredImg = np.expand_dims(flow2img(PredImg[0].permute(1, 2, 0).cpu().detach().numpy()),axis=0)
                elif "Mask" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W),mode='nearest')
                    PredImg = (PredImg.permute(0, 2, 3, 1).cpu().numpy()) * 255

                save_img = os.path.join(save_dir,key,"%06d.png"%t)
                if key == "PredImg" and not os.path.exists(os.path.dirname(save_img)):
                    os.makedirs(os.path.dirname(save_img))
                for tplus in range(0, bs):
                    PredImg_t = PredImg[tplus]
                    #PredImg_t = cv2.resize(PredImg[tplus],(output_W,output_H))
                    if "Mask" in key:
                        if t==0:
                            save_img = os.path.join(save_dir,"%s.png"%key)
                            cv2.imwrite(save_img, (PredImg[tplus]))
                    elif ("Composited" not in key and "PredImg" not in key):
                        if t==0:
                            save_img = os.path.join(save_dir,"%s.png"%key)
                            cv2.imwrite(save_img, (cv2.cvtColor(PredImg_t, cv2.COLOR_RGB2BGR)))
                    else:
                        save_img = os.path.join(save_dir,key,"%06d.png"%(t*bs+tplus))
                        cv2.imwrite(save_img, (cv2.cvtColor(PredImg_t, cv2.COLOR_RGB2BGR)))


    if DEBUG_TIME:
        for key, value in time_dict.items():
            print('avg %s:%.1f ms'%(key, value.avg*1000))
            print('sum %s:%.1f ms'%(key, value.sum*1000))
        print('count %d'%(time_dict['t_forward_batch'].count))
    for key in pred_imgs.keys():
        if ("BG" in key or "Z_" in key or "SmallSpeedAlpha" in key):
            os.system('rm -rf %s/%s'%(save_dir,key))
            continue
        """
        if os.path.exists(os.path.join(save_dir,key)):
            #os.system('convert -delay 3.33 %s/%s/*.png %s/%s_%s_%s.gif'%(save_dir, key, save_dir,key, name,opts.model_type))
            os.system('ffmpeg -loglevel quiet -framerate 30 -i %s/%s/%%06d.png %s/%s_%s_%s.gif -y'%(save_dir, key, save_dir,key, name,opts.model_type))
            #if key != "PredImg":
            #    os.system('rm -rf %s/%s'%(save_dir,key))
        """

