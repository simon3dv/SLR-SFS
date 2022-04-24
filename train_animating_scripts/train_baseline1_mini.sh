export DEBUG=0
export USE_SLURM=1

jobname=Baseline1Mini
folder=animating_synsin_20220323_bs16_eulerian_pconv_resnetencoder_resnetdecoder_softmax_baseline1_Mini
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:1 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=3dmr-head3d \
--cpus-per-task=32 \
python train_animating_unet.py --batch-size 2 --folder $folder --num_workers 32  \
        --resume --dataset 'eulerian_data' --use_inverse_depth --accumulation 'alphacomposite' \
        --model_type 'softmax_splating' --refine_model_type 'resnet_256W4UpDown64_de_resnet_nonorm' \
        --norm_G 'sync:spectral_batch' --gpu_ids 0 --render_ids 1 \
        --suffix '' --normalize_image --lr 0.0001 --train_Z  \
        --log-dir "logging/animating/%s/" --niter 100 --niter_decay 10 --use_softmax_splatter
