export DEBUG=0
export USE_SLURM=1

jobname=joint
folder=animating_synsin_20220511_joint_motion2ep200best_lrd2_nodecay_EPE0
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:8 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=3dmr-head3d \
--cpus-per-task=32 \
python train_animating_fixmotion.py --batch-size 16 --folder $folder --num_workers 16  \
        --resume --dataset 'eulerian_data_hint' --use_inverse_depth --accumulation 'wsum' \
        --model_type 'softmax_splating' --refine_model_type 'resnet_256W8UpDown64_de_resnet_pconv2' --pconv "pconv_pbn_woresbias" \
        --norm_G 'sync:spectral_batch' --gpu_ids 0,1,2,3,4,5,6,7 --render_ids 1 \
        --suffix '' --normalize_image --train_Z \
        --log-dir "logging/animating/%s/"  \
        --load_old_model --old_model "pretrained/baseline2_ep100_stage1ofSLRv1.pth" \
        --load_motion_regressor --motion_regressor_model "pretrained/motion2_ep200best.pth" \
        --train_motion --motion_model_type 'SPADE_unet_mask_motion' --motion_losses 1.0_EndPointError --div_flow 1.0 --lr_g 0.00025 --lr_d 0.001 --niter 100 --niter_decay 2 \
        --use_mask_as_motion_input --use_hint_as_motion_input --motion_norm_G 'sync:spectral_instance'

