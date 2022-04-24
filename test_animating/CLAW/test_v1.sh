export DEBUG=0
export USE_SLURM=1
SCENE=ALL
IMGDIR=data/CLAW/test
FLOWDIR=data/CLAW/test
PREDDIR=data/CLAW/results/20220406_N60_W768/eval_v1/v1
PRETRAINEDFILE="pretrained/SLR_v1_ep100stage1+ep50stage3.pth"
PYFILE="test_v1_4eval_rawsize.py"
ALIGNDATA="None"
#ALIGNDATA="data/CLAW/CLAW_align_max_frame_001_max600.json"
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:1 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=3dmr-head3d \
--cpus-per-task=8 \
python test_animating/CLAW/test_all_CLAW_scenes.py $IMGDIR $FLOWDIR $PREDDIR $PRETRAINEDFILE temp 768 60 1 $PYFILE $SCENE $ALIGNDATA $1 $2
