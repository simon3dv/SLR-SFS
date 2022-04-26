# SLR-SFS

## Simulating Fluids in Real-World Still Images 

Code release for the paper **Simulating Fluids in Real-World Still Images**

**Authors**: [Siming Fan](https://simon3dv.github.io/), [Jingtan Piao](#), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=en), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Hongsheng Li](http://www.ee.cuhk.edu.hk/~hsli/).

[[Paper](http://arxiv.org/abs/2204.11335)] [[Project Page](https://SimulatingFluids.github.io/)]

##### Our SLR sample (Still Input Image | Animated Video(480x256)):
<center class="half">
     <img src="https://user-images.githubusercontent.com/56864061/164973194-1d1d508b-af35-4802-aa18-bd0ad49af768.png" width="720"/><img src="https://user-images.githubusercontent.com/56864061/163476834-4fb912ec-4afc-4bb4-911c-97744cdb6ea6.gif" width="720"/>
</center>

<!--![SLR_input](https://user-images.githubusercontent.com/56864061/164972138-d466d728-8d9e-4f7f-85ed-0dba9e4650a3.jpg) ![SLR_sample](https://user-images.githubusercontent.com/56864061/163476834-4fb912ec-4afc-4bb4-911c-97744cdb6ea6.gif)--> 
##### Our SFS sample (Still Input Image | Animated Video(480x256)):
<center class="half">
     <img src="https://user-images.githubusercontent.com/56864061/164973201-a27f2cc6-6598-4a55-9f2f-d0b0379ef386.png" width="720"/><img src="https://user-images.githubusercontent.com/56864061/163478052-97f47118-8c73-423c-bede-4b014e99ea67.gif" width="720"/>
</center>
<!--![SLR_input](https://user-images.githubusercontent.com/56864061/164972151-05a21722-5634-4d2f-a8b7-b6dc297f6bf7.jpg) ![SFS_sample](https://user-images.githubusercontent.com/56864061/163478052-97f47118-8c73-423c-bede-4b014e99ea67.gif)-->

## Introduction
In this work, we tackle the problem of real-world fluid animation from a still image.  We propose a new and learnable representation, **surface-based layered representation(SLR)**, which decomposes the fluid and the static objects in the scene, to better synthesize the animated videos from a single fluid image. and design a **surface-only fluid simulation(SFS)** to model the evolution of the image fluids with better visual effects.

For more details of SLR-SFS, please refer to our paper and project page.

## Data preparetion
#### For evaluation:

Download [our CLAW test set(314.8MB)](https://drive.google.com/file/d/178y3M96fvYwmzZU-ngMosBBOrbJ-tDe-/view?usp=sharing) ([Description](https://slr-sfs.github.io/dataset/)) and put it to SLR-SFS/data/CLAW/test

#### For training:

Download [eulerian_data(Everything, 42.9GB)](https://drive.google.com/file/d/19f2PsKEaeAmspd1ceGkOEMhZsZNquZyF/view) ([Description](https://eulerian.cs.washington.edu/dataset/))

Generate mean video of ground truth for background training.(Need 4.8GB)
``` 
cd data
python average_gt_video.py
```

Download [our label(1MB)](https://drive.google.com/file/d/1WDvHbkVn-CwWXjCAZItBnUAmSvbfaed9/view) ([Description](https://slr-sfs.github.io/dataset/)) and put it to SLR-SFS/eulerian_data/fluid_region_rock_labels/all
.Make sure `opt.rock_label_data_path = "data/eulerian_data/fluid_region_rock_labels/all" in options/options.py` contains label file, scene w/o file is considered to have no rock in moving region. You can check tensorboard.


 

```
SLR-SFS
├── data
│   ├── eulerian_data
│   │   ├── train
│   │   ├── validation
│   │   ├── imageset_shallow.npy # list of videos containing transparent fluid in train/*.gt.mp4
│   │   │── align*.json # speed align information for gt motion in validation/*_motion.pth to avoid large invalid pixels after warping
│   │   ├── avr_image # containing mean image of each video in train/*.gt.mp4  
│   ├── eulerian_data.py #(for baseline training)
│   ├── eulerian_data_balanced1_mask.py #(for SLR training)
│   ├── eulerian_data_bg.py #(for BG training)
│   ├── CLAW # mentioned in paper
│   │   ├── test
│   │   ├── align*.json
│   ├── CLAWv2 # new collected test set with higher resolution and more diverse scenes(containing fountains, oceans, beaches, mists, etc., which is not included in CLAW_data)
│   │   ├── test
│   │   ├── align*.json
```

## Setup
```
conda create -n SLR python=3.9 -y
conda activate SLR
conda install -c pytorch pytorch==1.10.0 torchvision #pytorch/linux-64::pytorch-1.10.0-py3.9_cuda11.3_cudnn8.2.0_0

pip install tqdm opencv-python py-lz4framed matplotlib 
conda install cupy -c conda-forge
pip install lpips# for evaluation
pip install av tensorboardX tensorboard # for training
```

## Inference
Download our pretrained model mentioned in Table 1,2 of the paper:

[baseline2](https://drive.google.com/file/d/1eWLh23IkSRtmA1lyZGjivqC97DK3wDEu/view?usp=sharing) (Modified Holynski(Baseline)): 100epochs + (lower learning rate)50epochs

[Ours_stage_1](https://drive.google.com/file/d/1WyPZ2GHera0RIl5MTa5lHyh21oK-IlG1/view?usp=sharing) (Ours(Stage 1)): 100epochs

[Ours_stage_2](https://drive.google.com/file/d/1Blu-G-zERrcpk3J3sGWuawhcK8dLFG1x/view?usp=sharing) (Background Only, used in Ours(Stage 2)): 100epochs

[Ours_v1](https://drive.google.com/file/d/1WzqWB-a85hZDhdos2CWezqT95IjJXzVN/view?usp=sharing) (Ours): 100epochs baseline2 + 100epochs BG + (lower learning rate)50 epochs Ours

1.For evaluation under gt motion:
```Ours CLAW testset
# For our v1 model, 60 frames, gt motion(Table 1 in the paper)
bash test_animating/CLAW/test_v1.sh
bash evaluation/eval_animating_CLAW.sh
# For baseline2 model, 60 frames, gt motion(Table 1 in the paper)
bash test_animating/CLAW/test_baseline2.sh
bash evaluation/eval_animating_CLAW.sh
## You can also use sbatch script test_animating/test_sbatch_2.sh
```


2.You can also use aligned gt motion to avoid large holes for better animation:
```
bash test_animating/CLAW/test_v1_align.sh
```
Results will be the same as:

3.Run with smaller resolution by replacing 768 to 256 in test_animating/CLAW/test_v1_align.sh, etc.


## Training
To train baseline model under gt motion, run the following scripts
```
# For baseline training
bash train_animating_scripts/train_baseline1.sh
# For baseline2 training (w/ pconv)
bash train_animating_scripts/train_baseline2_pconv.sh
```

To train our SLR model under gt motion, run the following scripts
```
# Firstly, train Surface Fluid Layer for 100 epochs
bash train_animating_scripts/train_baseline2_pconv.sh
# Secondly, generate "mean video" and train Background Layer for 100 epochs
bash train_animating_scripts/train_bg.sh
# Lastly, unzip the label file to proper directory, train alpha,  finetune Fluid and BG.
bash train_alpha_finetuneBG_finetuneFluid_v1.sh
(Notice: check tensorboard to see whether your groundtruth alpha is right)
```



## ToDo list
- [x] pretrained model and code of our reproduced Holynski's method(w/o motion estimation)
- [x] pretrained model and code of SLR and pretrained model
- [ ] code of SFS
- [ ] CLAWv2 testset
- [ ] pretrained model and code of motion estimation from single image


## Citation
If you find this work useful in your research, please consider cite:
```
@article{fan2022SLR,
  author    = {Siming Fan, Jingtan Piao, Chen Qian, Kwan-Yee Lin, Hongsheng Li},
  title     = {Simulating Fluids in Real-World Still Images},
  journal   = {arXiv preprint},
  volume    = {arXiv:2204.11335},
  year      = {2022},
}
```
               


