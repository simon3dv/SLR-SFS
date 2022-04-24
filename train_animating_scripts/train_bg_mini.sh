export DEBUG=0
export USE_SLURM=1

jobname=BG1
folder=animating_synsin_20220406_bs16_BG1_mini
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:8 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=3dmr-head3d \
--cpus-per-task=16 \
python train_animating_BG.py --batch-size 16 --folder $folder --num_workers 16 --model_type "bg" \
        --resume --dataset 'eulerian_data_bg' --use_inverse_depth --accumulation 'wsum' --bg_refine_model_type 'resnet_256W5UpDown64BG_nonorm' --ngf 64 \
        --norm_G 'sync:spectral_batch' --gpu_ids 0,1,2,3,4,5,6,7 --render_ids 1 \
        --suffix '' --normalize_image --lr 0.0001 --MVloss 1.0 \
        --log-dir "logging/bg/%s/" --W 256 --train_bg
