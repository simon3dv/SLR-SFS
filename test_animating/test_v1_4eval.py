import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(os.path.dirname(BASE_DIR))#,'../')
sys.path.append(ROOT_DIR)
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel
from options.options import get_model
from PIL import Image
import cv2
#from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import torch.nn.functional as F
from models.projection.euler_integration_manipulator import EulerIntegration
import time
import pickle
import lz4framed
import json
from utils.flow_utils import flow2img
#from gpu_memory_log import gpu_memory_log
os.environ['DEBUG'] = '0'
torch.backends.cudnn.enabled = True

DEBUG_TIME = False
MODEL_NAMES = [
    'AnimatingModel',
]
bs = 1

def angle2rad(angle):
    return angle*np.pi/180
def rad2angle(rad):
    return rad*180/np.pi

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(np.frombuffer(buffer=strFlow, dtype=np.float32, count=1, offset=0) == 202021.25)

    intWidth = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=4)[0]
    intHeight = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=8)[0]

    return np.frombuffer(buffer=strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])


def load_compressed_tensor(filename):
    retval = None
    with open(filename, mode='rb') as file:
        retval = torch.from_numpy(pickle.loads(lz4framed.decompress(file.read())))
    return retval

if __name__=='__main__':
    rootdir = ""
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        image_file = os.path.join(rootdir,
                                  "data/000031.jpg")
    if len(sys.argv) > 2:
        flow_file = sys.argv[2]
    else:
        flow_file = os.path.join(rootdir,"data/000031.flo")
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    else:
        save_dir = os.path.join(rootdir, "warp")

    if len(sys.argv) > 4:
        warp_pretrained_model = sys.argv[4]
    else:
        warp_pretrained_model = "pretrained/model_epoch.pthbest100"

    if len(sys.argv) > 5:
        name = sys.argv[5]
    else:
        name = "Demo"

    if len(sys.argv) > 6:
        W = int(sys.argv[6])
    else:
        W = 768

    if len(sys.argv) > 7:
        N = int(sys.argv[7])
    else:
        N = 150

    if len(sys.argv) > 8:
        speed = float(sys.argv[8])
    else:
        speed = 0.25

    if len(sys.argv) > 9:
        align_data = sys.argv[9]
    else:
        align_data = "None"
    
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
    output_W = output_W // 2
    output_H = output_H // 2
    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((opts.W, opts.W)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)

    flows_f = []
    flows_p = []

    if ".pth" in flow_file:
        flow = load_compressed_tensor(flow_file)
    elif ".flo" in flow_file:
        flow = torch.FloatTensor(read_flo(flow_file))
        flow = flow.permute((2, 0, 1)).contiguous().unsqueeze(0)
    print("Mean flow : %.2f"%flow.abs().mean())
    flow_scale = [opts.W / flow.shape[3] * speed, opts.W / flow.shape[2] * speed]
    flow = flow * torch.FloatTensor(flow_scale).view(1,2,1,1)
    minn = min(flow.shape[3], flow.shape[2])
    flow = F.interpolate(flow, (opts.W, opts.W))
    flow = torch.FloatTensor(flow)
    flow = flow.view(1, 2, -1)  # (2, WW)


    img = img.unsqueeze(0).cuda()
    flow = flow.cuda()
    euler_integration = EulerIntegration(opts)
    model_to_test.model.module.eval()
    with torch.no_grad():
        # encoder
        start_fs = model_to_test.model.module.encoder(img)
        bg_img = model_to_test.model.module.net_bg(img)

        if align_data != "None":
            with open(align_data, "r") as f:
                align_dict = json.load(f)
            frame = align_dict[name]
            flow = flow * frame / FRAME

        batch = {"features": [start_fs],
                 "BGImg":[bg_img],
                 "images":[img],
                 'motions': [flow.view(1,2,opts.W,opts.W)],
                 #"depths": [torch.ones(img[:, 0:1, :, :].shape)+ torch.rand(img[:, 0:1, :, :].shape) * 1e-4],
                 'index': None}
        for t in tqdm(range(0, FRAME//bs)):
            batch['index'] = torch.Tensor([0, t*bs, FRAME - 1]).long().view(1,3)
            for tplus in range(1, bs):
                batch['index'] = torch.cat((batch['index'],torch.Tensor([0, t*bs+tplus, FRAME - 1]).long().view(1,3)), 0)

            with torch.no_grad():
                pred_imgs = model_to_test.model.module.forward_flow(batch)
            for key in pred_imgs.keys():
                if "Img" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W), mode='bilinear')
                    PredImg = (PredImg.permute(0,2,3,1).cpu().numpy() * 0.5 + 0.5) * 255 # bs, 3, 672, 378
                elif "Alpha" in key:#or "Z_" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W), mode='bilinear')
                    PredImg = (PredImg.permute(0,2,3,1).cpu().numpy()) * 255
                elif "Z_" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W), mode='bilinear')
                    PredImg = (PredImg - PredImg.min()) / (PredImg.max()-PredImg.min())
                    PredImg = (PredImg.permute(0, 2, 3, 1).cpu().numpy()) * 255
                elif "Motion" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W), mode='bilinear')
                    PredImg = np.expand_dims(flow2img(PredImg[0].permute(1, 2, 0).cpu().detach().numpy()), axis=0)
                elif "Mask" in key:
                    PredImg = F.interpolate(pred_imgs[key], (output_H, output_W), mode='nearest')
                    PredImg = (PredImg.permute(0, 2, 3, 1).cpu().numpy()) * 255
                save_img = os.path.join(save_dir,key,"%06d.png"%t)
                if not os.path.exists(os.path.dirname(save_img)):
                    os.makedirs(os.path.dirname(save_img))
                for tplus in range(0, bs):
                    PredImg_t = PredImg[tplus]
                    #PredImg_t = cv2.resize(PredImg[tplus],(output_W,output_H))
                    if "Mask" in key:
                        if t==0:
                            save_img = os.path.join(save_dir,"%s.png"%key)
                            cv2.imwrite(save_img, PredImg_t)
                    elif ("BG" in key or "Z_" in key or "SmallSpeedAlpha" in key):
                        if t==0:
                            save_img = os.path.join(save_dir,"%s.png"%key)
                            if "Alpha" in key or "Z_f" in key:
                                cv2.imwrite(save_img, PredImg_t)
                            else:
                                cv2.imwrite(save_img, (cv2.cvtColor(PredImg_t, cv2.COLOR_RGB2BGR)))
                    else:
                        save_img = os.path.join(save_dir,key,"%06d.png"%(t*bs+tplus))
                        if "Alpha" in key:
                            cv2.imwrite(save_img, PredImg_t)
                        else:
                            cv2.imwrite(save_img, (cv2.cvtColor(PredImg_t, cv2.COLOR_RGB2BGR)))


    for key in pred_imgs.keys():
        if ("BG" in key or "Z_" in key or "SmallSpeedAlpha" in key):
            os.system('rm -rf %s/%s'%(save_dir,key))
            continue
        if os.path.exists(os.path.join(save_dir,key)):
            #os.system('convert -delay 3.33 %s/%s/*.png %s/%s_%s_%s.gif'%(save_dir, key, save_dir,key, name,opts.model_type))
            os.system('ffmpeg -loglevel quiet -framerate 30 -i %s/%s/%%06d.png -framerate 30 %s/%s_%s_%s.gif -y'%(save_dir, key , save_dir, key, name,opts.model_type))
            #if key != "PredImg": #and key!="CompositeFluidAlpha":
            #    os.system('rm -rf %s/%s'%(save_dir,key))
