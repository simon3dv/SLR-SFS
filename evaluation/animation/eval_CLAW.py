import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#BASE_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
import av
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
import torchvision.transforms as transforms
import numpy as np
import pickle
import lz4framed
from PIL import Image
import time
import random
import torch.nn.functional as F
import torchvision
from evaluation.animation.metrics import perceptual_sim, psnr, ssim_metric
from models.networks.pretrained_networks import PNet
import lpips
import json
loss_fn_alex = lpips.LPIPS(net='alex',version='0.1').cuda()
from models.base_model import BaseModel
from models.networks.sync_batchnorm import convert_model
from options.options import get_dataset, get_model
from options.test_options import ArgumentParser
import tqdm
import glob
from utils.utils import load_compressed_tensor, VideoReader
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

def compute_perceptual_similarity(p_img, t_img):
    t_perc_sim = perceptual_sim(p_img, t_img, vgg16).item()
    t_perc_lpips = loss_fn_alex(p_img, t_img).item()
    t_ssim = ssim_metric(p_img, t_img).item()
    t_psnr = psnr(p_img, t_img).item()
    return {
        "LPIPS": t_perc_lpips,
        "Perceptual": t_perc_sim,
        "PSNR": t_psnr,
        "SSIM": t_ssim,
    }



if __name__ == "__main__":
    # Load VGG16 for feature similarity
    vgg16 = PNet().to("cuda")
    vgg16.eval()
    vgg16.cuda()

    #osize = (360, 640)
    if len(sys.argv) > 1:
        pred_dir = sys.argv[1]
    else:
        pred_dir = ""
    if len(sys.argv) > 2:
        gt_dir = sys.argv[2]
    else:
        gt_dir = ""

    save_json = os.path.join(pred_dir,"../metric.json")
    video_dir = sorted([x for x in os.listdir(pred_dir)
                        if os.path.exists(os.path.join(pred_dir,x,"PredImg/000000.png"))
                        and (x+".mp4") in os.listdir(gt_dir)
                        and len(os.listdir(os.path.join(pred_dir,x,"PredImg")))==60])
    print(len(video_dir))
    for i_pred, pred_name in enumerate(os.listdir(pred_dir)):
        if pred_name not in video_dir:
            print("do not eval %s, %d,%d,%d"%(pred_name,
                                              os.path.exists(os.path.join(pred_dir,pred_name,"PredImg/000000.png")),
                                              (pred_name+".mp4") in os.listdir(gt_dir),
                                              len(os.listdir(os.path.join(pred_dir, pred_name, "PredImg"))) == 60
                                              ))

    values_lpip = []
    values_perc = []
    values_ssim = []
    values_psnr = []
    results_dict = {
        'TotalLPIPS': {}, 'TotalPerceptual': {}, 'TotalPSNR': {}, 'TotalSSIM': {},
        'TotalLPIPS_std': {}, 'TotalPerceptual_std': {}, 'TotalPSNR_std': {}, 'TotalSSIM_std': {},
        'LPIPS':{},'Perceptual':{},'PSNR':{},'SSIM':{},
        'LPIPS_std': {}, 'Perceptual_std': {}, 'PSNR_std': {}, 'SSIM_std': {},}
    for i_video, video in enumerate(video_dir):
        print(video)
        values_video_lpip = []
        values_video_perc = []
        values_video_ssim = []
        values_video_psnr = []
        imgdir = os.path.join(pred_dir,video,"PredImg")
        gtvideo = os.path.join(gt_dir,"%s.mp4"%video)
        video_reader = VideoReader(gtvideo, None, "mp4")
        for i_image in range(60):
            pred_img = Image.open(os.path.join(imgdir, "%06d.png"%i_image))
            pred_img = transforms.ToTensor()(pred_img)#(3,H,W)
            gt_img = torch.FloatTensor(video_reader[i_image])[0].permute(2, 0, 1) / 255.0#(1, 720, 1280, 3) uint8
            gt_img = transforms.ToPILImage()(gt_img)
            gt_img = transforms.Resize(pred_img.shape[1:3], Image.BILINEAR)(gt_img)
            gt_img = transforms.ToTensor()(gt_img)#3,360,640

            metric = compute_perceptual_similarity(pred_img.unsqueeze(0).cuda(), gt_img.unsqueeze(0).cuda())
            values_video_lpip.append( metric["LPIPS"])
            values_video_perc.append( metric["Perceptual"])
            values_video_ssim.append( metric["SSIM"])
            values_video_psnr.append( metric["PSNR"])

            values_lpip.append( metric["LPIPS"])
            values_perc.append( metric["Perceptual"])
            values_ssim.append( metric["SSIM"])
            values_psnr.append( metric["PSNR"])
        avg_video_perclpips = np.mean(np.array(values_video_lpip))
        std_video_perclpips = np.std(np.array(values_video_lpip))

        avg_video_percsim = np.mean(np.array(values_video_perc))
        std_video_percsim = np.std(np.array(values_video_perc))

        avg_video_psnr = np.mean(np.array(values_video_psnr))
        std_video_psnr = np.std(np.array(values_video_psnr))

        avg_video_ssim = np.mean(np.array(values_video_ssim))
        std_video_ssim = np.std(np.array(values_video_ssim))


        results_dict['LPIPS'][video] = avg_video_perclpips
        results_dict['Perceptual'][video] = avg_video_percsim
        results_dict['SSIM'][video] = avg_video_ssim
        results_dict['PSNR'][video] = avg_video_psnr
        results_dict['LPIPS_std'][video] = std_video_perclpips
        results_dict['Perceptual_std'][video] = std_video_percsim
        results_dict['SSIM_std'][video] = std_video_ssim
        results_dict['PSNR_std'][video] = std_video_psnr


    avg_perclpips = np.mean(np.array(values_lpip))
    std_perclpips = np.std(np.array(values_lpip))

    avg_percsim = np.mean(np.array(values_perc))
    std_percsim = np.std(np.array(values_perc))

    avg_psnr = np.mean(np.array(values_psnr))
    std_psnr = np.std(np.array(values_psnr))

    avg_ssim = np.mean(np.array(values_ssim))
    std_ssim = np.std(np.array(values_ssim))

    results_dict['TotalLPIPS'] = avg_perclpips
    results_dict['TotalPerceptual'] = avg_percsim
    results_dict['TotalSSIM'] = avg_ssim
    results_dict['TotalPSNR']= avg_psnr
    results_dict['TotalLPIPS_std'] = std_perclpips
    results_dict['TotalPerceptual_std'] = std_percsim
    results_dict['TotalSSIM_std'] = std_ssim
    results_dict['TotalPSNR_std'] = std_psnr

    with open(save_json, "w") as f:
        json.dump(results_dict, f)