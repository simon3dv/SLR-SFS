export DEBUG=0
export USE_SLURM=1
PREDDIR=data/CLAW/results/20220406_N60_W768/eval_v1/v1
GTDIR=data/CLAW/test
jobname=eval
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:1 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=pat_taurus \
--cpus-per-task=12 \
python evaluation/animation/eval_CLAW.py $PREDDIR $GTDIR
