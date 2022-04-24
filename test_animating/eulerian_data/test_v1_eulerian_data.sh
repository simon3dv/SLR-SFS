export DEBUG=0
export USE_SLURM=1
SCENE=ALL
IMGDIR=/mnt/lustre/fansiming/project/SLR-SFS/data/eulerian_data/validation
FLOWDIR=/mnt/lustre/fansiming/project/SLR-SFS/data/eulerian_data/validation
PREDDIR=/mnt/lustre/fansiming/project/SLR-SFS/data/eulerian_data/results/20220406_N60_W768/eval_v13/v13
PRETRAINEDFILE="pretrained/SLR_model/model_epoch.pthbestperc"
PYFILE="test_v1_4eval.py"
ALIGNDATA="None"
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:1 \
-n1 --ntasks-per-node=1 \
--job-name=test \
--kill-on-bad-exit=1 \
--partition=3dmr-head3d \
--cpus-per-task=8 \
python test_animating/eulerian_data/test_all_eulerian_data_val_scenes.py $IMGDIR $FLOWDIR $PREDDIR $PRETRAINEDFILE temp 768 60 1 $PYFILE $SCENE $ALIGNDATA $1 $2
