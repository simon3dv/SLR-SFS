export DEBUG=0
export USE_SLURM=1
jobname=motion19
folder=20220509motion_bs16_default_EPE_motion19
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:4 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=pat_taurus \
--cpus-per-task=32 \
python train_motion_unet.py --batch-size 16 --folder $folder --num_workers 32  \
        --resume --dataset 'eulerian_data_motion_hint' --use_inverse_depth --accumulation 'alphacomposite' --use_online_hint \
        --model_type 'SPADE_unet_mask_motion' --refine_model_type 'resnet_256W8UpDown64Motion' --train_motion \
        --norm_G 'sync:spectral_batch' --motion_norm_G "sync:spectral_instance" --gpu_ids 0,1,2,3 --render_ids 1  --div_flow 1.0 \
        --suffix '' --lr 0.0001 --normalize_image --output_nc 2 --motion_losses 10.0_EndPointError --discriminator_losses pix2pixHD --motionH 256 --motionW 256 \
        --use_mask_as_motion_input --use_hint_as_motion_input --niter 1000 --niter_decay 2 --log-dir "logging/motion/%s/"
