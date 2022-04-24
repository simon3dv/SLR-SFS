export DEBUG=0
export USE_SLURM=1

jobname=Sepv13
folder=animating_synsin_20220406_bs16_SeperateJoint13
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun \
--mpi=pmi2 \
--gres=gpu:8 \
-n1 --ntasks-per-node=1 \
--job-name=$jobname \
--kill-on-bad-exit=1 \
--partition=pat_taurus \
--cpus-per-task=32 \
python train_animating_alpha_2layers_joint_finetuneBGFluid_L1.py --batch-size 16 --folder $folder --num_workers 16  \
        --resume --dataset 'eulerian_data_balanced1_mask' --use_inverse_depth --accumulation 'wsum' --bg_refine_model_type 'resnet_256W8UpDown64BG_nonorm' \
        --alpha_refine_model_type 'resnet_256W8UpDown64Layers_de_resnet_pconv2_nonorm' \
        --model_type 'softmax_splating_2layers_alpha_seperate' --refine_model_type 'resnet_256W8UpDown64_de_resnet_pconv2_nonorm' --pconv "pconv_pbn_woresbias" --out_channel 65 --ngf 64 \
        --norm_G 'sync:spectral_batch' --gpu_ids 0,1,2,3,4,5,6,7 --render_ids 1 \
        --suffix '' --normalize_image --lr 0.0001 --train_Z --lr_g 0.00025 --lr_d 0.001 \
        --log-dir "logging/animating/%s/" --W 256 --ATVloss 0.3  --AKLloss 0.0 --ADCloss 1.0 --FluidRegionloss 3 --balanced_weight 10  --RockRegionloss 30 --RockRegionlossDecay 20 \
        --load_old_model --old_model "/mnt/lustre/fansiming/project/mysynsin/pretrain/animating_synsin_20220104_bs16_eulerian_pconv_resnetencoder_resnetdecoder_softmax_Z/model_epoch.pthbest100" \
        --load_bg_model --bg_model "/mnt/lustre/fansiming/project/mysynsin/logging/bg/eulerian_data_bg/2022-02/animating_synsin_20220207_bs16_BG1/models/lr0.00010_bs16_modelbg_splxyblending/noise_bnsync:spectral_batch_refresnet_256W8UpDown64Layers_de_resnet_pconv2_nonorm_dunet_camxysFalse|False/_init_databoth_seed0/_multiFalse_losses1.0|l110.0|content_izFalse_alphaFalse__vol_ganpix2pixHD/model_epoch.pthbest" \
        --train_bg --train_alpha --MVloss 1.0   --niter 2 --niter_decay 20 --RockRegionlosstarget 0.25 --use_alpha0_as_blending_weight

