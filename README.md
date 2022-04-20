# SLR-SFS

## Simulating Fluids in Real-World Still Images 

Code release for the paper **Simulating Fluids in Real-World Still Images**

**Authors**: [Siming Fan](https://simon3dv.github.io/), [Jingtan Piao](#), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Chen Qian](https://scholar.google.com/citations?user=AerkT0YAAAAJ&hl=en), [Hongsheng Li](http://www.ee.cuhk.edu.hk/~hsli/).

[[Paper](#)] [[Project Page](https://SimulatingFluids.github.io/)]

#####Our SLR sample:

![SLR_sample](https://user-images.githubusercontent.com/56864061/163476834-4fb912ec-4afc-4bb4-911c-97744cdb6ea6.gif) 

#####Our SFS sample:

![SFS_sample](https://user-images.githubusercontent.com/56864061/163478052-97f47118-8c73-423c-bede-4b014e99ea67.gif)

## Introduction
In this work, we tackle the problem of real-world fluid animation from a still image.  We propose a new and learnable representation, **surface-based layered representation(SLR)**, which decomposes the fluid and the static objects in the scene, to better synthesize the animated videos from a single fluid image. and design a **surface-only fluid simulation(SFS)** to model the evolution of the image fluids with better visual effects.

For more details of SLR-SFS, please refer to our paper and project page.

## Data preparetion
#####For evaluation:

Download [our CLAW test set(314.8MB)](https://drive.google.com/file/d/178y3M96fvYwmzZU-ngMosBBOrbJ-tDe-/view?usp=sharing) ([Description](https://slr-sfs.github.io/dataset/)) and put it to SLR-SFS/data/CLAW_data/test

####For training:

Download [eulerian_data(Everything, 42.9GB)](https://drive.google.com/file/d/19f2PsKEaeAmspd1ceGkOEMhZsZNquZyF/view) ([Description](https://eulerian.cs.washington.edu/dataset/))

Download [our label(1MB)](https://drive.google.com/file/d/1WDvHbkVn-CwWXjCAZItBnUAmSvbfaed9/view) ([Description](https://slr-sfs.github.io/dataset/)) and put it to SLR-SFS/eulerian_data/fluid_region_rock_labels/all
.Make sure `opt.rock_label_data_path = "data/eulerian_data/fluid_region_rock_labels/all" in options/options.py` contains label file, scene w/o file is considered to have no rock in moving region.


 
## ToDo list
- [x] dataset and addtional label release
- [ ] pretrained model and code of our reproduced Holynski's method
- [ ] pretrained model and code of SLR and pretrained model
- [ ] code of SFS
- [ ] pretrained model and code of motion estimation from single image

## Citation
If you find this work useful in your research, please consider cite:
```
@article{fan2022SLR,
  author    = {Siming Fan, Jingtan Piao, Kwan-Yee Lin, Chen Qian, Hongsheng Li},
  title     = {Simulating Fluids in Real-World Still Images},
  journal   = {arxiv},
  year      = {2022},
}
``` 


