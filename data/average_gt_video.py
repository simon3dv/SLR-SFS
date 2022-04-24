import numpy as np
from PIL import Image
import os
import json
import sys
PATH = os.getcwd()
sys.path.append(os.path.join(PATH))
from utils.utils import VideoReader, load_compressed_tensor

if __name__=='__main__':
	datadir = os.path.join("data/eulerian_data")
	save_name = "avr_image"
	save_dir = os.path.join(datadir,save_name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for dataset in ["train", "validation"]:
		imageset = np.array([x[:11] for x in os.listdir(os.path.join(datadir,dataset)) if 'mp4' in x])
		for scene_i, scene in enumerate(sorted(imageset)):
			video_file = os.path.join(datadir, dataset, "%s_gt.mp4"%(scene))
			save_path =  os.path.join(datadir, "%s/%s.png"%(save_name,scene))
			video_reader = VideoReader(video_file, None, "mp4")
			N = len(video_reader)
			mean_image = video_reader[0].astype(np.float32)
			for i in range(1,N):
				mean_image += video_reader[i].copy()
			mean_image = mean_image / N
			im = Image.fromarray(mean_image[0].astype(np.uint8))
			im.save(save_path)
