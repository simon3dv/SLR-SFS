export DEBUG=0
export USE_SLURM=0

folder=animating_synsin_20220323_bs16_eulerian_pconv_resnetencoder_resnetdecoder_softmax_baseline1_Mini
python train_animating.py --batch-size 2 --folder $folder --num_workers 32  \
        --resume --dataset 'eulerian_data' --use_inverse_depth --accumulation 'alphacomposite' \
        --model_type 'softmax_splating' --refine_model_type 'resnet_256W4UpDown64_de_resnet_nonorm' \
        --norm_G 'sync:spectral_batch' --gpu_ids 0 --render_ids 1 \
        --suffix '' --normalize_image --lr 0.0001 --train_Z  \
        --log-dir "logging/animating/%s/" --niter 100 --niter_decay 10 --use_softmax_splatter
