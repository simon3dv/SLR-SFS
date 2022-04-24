#!/bin/bash
#SBATCH --partition=3dmr-head3d
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
export DEBUG=0
export USE_SLURM=1
SCENE=ALL
IMGDIR=data/CLAW/test
FLOWDIR=data/CLAW/test
PREDDIR=data/CLAW/results/20220406_N60_W768/eval_v1/v1
PRETRAINEDFILE="pretrained/SLR_v1_ep100stage1+ep50stage3.pth"
PYFILE="test_v1_4eval_rawsize.py"
ALIGNDATA="None"
#ALIGNDATA="/mnt/lustre/fansiming/project//data/mydataset/gtmotion_warp_outputs/mydataset_align_max_frame_001_bin_max600.json"
python test_animating/CLAW/test_all_CLAW_scenes.py $IMGDIR $FLOWDIR $PREDDIR $PRETRAINEDFILE temp 768 60 1 $PYFILE $SCENE $ALIGNDATA $1 $2
