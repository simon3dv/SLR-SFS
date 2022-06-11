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
PREDDIR=data/CLAW/results/20220610_N60_W768/eval_joint20m19fixm_ep50_th_retrain/joint20m19fixm
# w/ Finetune
PRETRAINEDFILE="pretrained/baseline2+motion2+fixedMotionFinetune.pth"
# w/o Finetune
# PRETRAINEDFILE="pretrained/baseline2+motion2.pth"
PYFILE="test_motion_4eval_rawsize_threshold.py"
ALIGNDATA="None"
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:1 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=3dmr-head3d \
--cpus-per-task=8 \
python test_animating/CLAW/test_all_CLAW_scenes.py $IMGDIR $FLOWDIR $PREDDIR $PRETRAINEDFILE temp 768 60 1 $PYFILE $SCENE $ALIGNDATA $1 $2
