import argparse
import datetime
import os
import time


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="self-supervised view synthesis"
        )
        self.add_data_parameters()
        self.add_train_parameters()
        self.add_model_parameters()

    def add_model_parameters(self):
        model_params = self.parser.add_argument_group("model")
        model_params.add_argument(
            "--model_type",
            type=str,
            default="zbuffer_pts",
            choices=(
                "zbuffer_pts",
                "zbuffer_meshes",
                "deepvoxels",
                "viewappearance",
                "tatarchenko",
                "softmax_splating",
                "bg",
                "alpha",
                "rock_seg",
                "moving_region_rock_seg",
                "softmax_splating_rawsize",
                "softmax_splating_Zmax",
                "softmax_splating_Zsum",
                "softmax_splating_mesh",
                "softmax_splating_alpha_encoder",
                "softmax_splating_alpha_encoder_2decoder",
                "softmax_splating_2layers_alpha",
                "softmax_splating_2layers_singlealpha_seperate",
                "softmax_splating_2layers_alpha_seperate",
                "softmax_splating_2layers_rockseg",
                "softmax_splating_joint",
                "unet_motion",
                "unet_mask_motion",
                "SPADE_unet_motion",
                "SPADE_unet_mask_motion",
                "pix2pixHD_motion",
                "pix2pixHDorigin_motion",
                "pix2pixHD_synsin_motion",
                "animating3d_cycle"
            ),
            help='Model to be used.'
        )
        model_params.add_argument(
            "--refine_model_type", type=str, default="resnet_256W8UpDown64Layers_de_resnet_pconv2_nonorm",
            help="Model to be used for the refinement network and the feature encoder."
        )
        model_params.add_argument(
            "--bg_refine_model_type", type=str, default="resnet_256W8UpDown64BG_nonorm",
        )
        model_params.add_argument(
            "--motion_refine_model_type", type=str, default="resnet_256W8UpDown64Motion_nonorm",
        )
        model_params.add_argument(
            "--alpha_refine_model_type", type=str, default="resnet_256W8UpDown64Alpha_nonorm",
        )
        model_params.add_argument(
            "--motion_model_type", type=str, default="unet",
            help="Model to be used for the motion estimation network."
        )
        model_params.add_argument(
            "--accumulation",
            type=str,
            default="wsum",
            choices=("wsum", "wsumnorm", "alphacomposite"),
            help="Method for accumulating points in the z-buffer. Three choices: wsum (weighted sum), wsumnorm (normalised weighted sum), alpha composite (alpha compositing)"
        )

        model_params.add_argument(
            "--depth_predictor_type",
            type=str,
            default="unet",
            choices=("unet", "hourglass", "true_hourglass"),
            help='Model for predicting depth'
        )
        model_params.add_argument(
            "--splatter",
            type=str,
            default="xyblending",
            choices=("xyblending"),
        )
        model_params.add_argument("--rad_pow", type=int, default=2,
            help='Exponent to raise the radius to when computing distance (default is euclidean, when rad_pow=2). ')
        model_params.add_argument("--num_views", type=int, default=2,
            help='Number of views considered per input image (inlcluding input), we only use num_views=2 (1 target view).')
        model_params.add_argument(
            "--crop_size",
            type=int,
            default=256,
            help="Crop to the width of crop_size (after initially scaling the images to load_size.)",
        )
        model_params.add_argument(
            "--aspect_ratio",
            type=float,
            default=1.0,
            help="The ratio width/height. The final height of the load image will be crop_size/aspect_ratio",
        )
        model_params.add_argument(
            "--norm_D",
            type=str,
            default="spectralinstance",
            help="instance normalization or batch normalization",
        )
        model_params.add_argument(
            "--noise", type=str, default="", choices=("style", "")
        )
        model_params.add_argument(
            "--learn_default_feature", action="store_true", default=True
        )
        model_params.add_argument(
            "--use_camera", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_focal_loss", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_sum1_alpha", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_fluid_alpha_only", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_bg_alpha_only", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_bg_as_alpha_input", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_fluid_as_alpha_input", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_mask_as_alpha_input", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_alpha0_as_blending_weight", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_alpha_softmax", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_pretrained_backbone", action="store_true", default=False
        )
        model_params.add_argument("--pp_pixel", type=int, default=128,
            help='K: the number of points to conisder in the z-buffer.'
        )
        model_params.add_argument("--tau", type=float, default=1.0,
            help='gamma: the power to raise the distance to.'
        )
        model_params.add_argument(
            "--use_gt_depth", action="store_true", default=False
        )
        model_params.add_argument(
            "--train_depth", action="store_true", default=False
        )
        model_params.add_argument(
            "--smooth_depth", type=float, default=0.0,
        )
        model_params.add_argument(
            "--train_motion", action="store_true", default=False
        )
        model_params.add_argument(
            "--train_bg", action="store_true", default=False
        )
        model_params.add_argument(
            "--train_alpha", action="store_true", default=False
        )
        model_params.add_argument(
            "--train_inpaint", action="store_true", default=False
        )
        model_params.add_argument(
            "--train_inpaint_gan", action="store_true", default=False
        )
        model_params.add_argument(
            "--misc_warping", action="store_true", default=False
        )
        model_params.add_argument(
            "--only_high_res", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_inverse_depth", action="store_true", default=False,
            help='If true the depth is sampled as a long tail distribution, else the depth is sampled uniformly. Set to true if the dataset has points that are very far away (e.g. a dataset with landscape images, such as KITTI).'
        )
        model_params.add_argument(
            "--ndf",
            type=int,
            default=64,
            help="# of discrim filters in first conv layer",
        )
        model_params.add_argument(
            "--use_xys", action="store_true", default=False
        )
        model_params.add_argument(
            "--output_nc",
            type=int,
            default=3,
            help="# of output image channels",
        )
        model_params.add_argument("--norm_G", type=str, default="batch")
        model_params.add_argument("--motion_norm_G", type=str, default="spectral_instance")
        model_params.add_argument(
            "--ngf",
            type=int,
            default=64,
            help="# of gen filters in first conv layer",
        )
        model_params.add_argument(
            "--radius",
            type=float,
            default=4,
            help="Radius of points to project",
        )
        model_params.add_argument(
            "--voxel_size", type=int, default=64, help="Size of latent voxels"
        )
        model_params.add_argument(
            "--num_upsampling_layers",
            choices=("normal", "more", "most"),
            default="normal",
            help="If 'more', adds upsampling layer between the two middle resnet blocks. "
            + "If 'most', also add one more upsampling + resnet layer at the end of the generator",
        )
        model_params.add_argument(
            "--blur_radius",
            type=float,
            default=1e-8,
            help=" Float distance in the range [0, 2] used to expand the face bounding boxes for rasterization. Setting blur radius results in blurred edges around the shape instead of a hard boundary. Set to 0 for no blur.",
        )
        model_params.add_argument(
            "--faces_per_pixel",
            type=int,
            default=1,
            help="Number of faces to save per pixel, returning the nearest faces_per_pixel points along the z-axis.",
        )
        model_params.add_argument(
            "--use_local_G", action="store_true", default=False
        )
        model_params.add_argument(
            "--use_beta", action="store_true", default=False
        )

    def add_data_parameters(self):
        dataset_params = self.parser.add_argument_group("data")
        dataset_params.add_argument("--dataset", type=str, default="kitti_depth")
        dataset_params.add_argument(
            "--use_semantics", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--config",
            type=str,
            default="../opt/habitat-api/configs/tasks/pointnav_rgbd.yaml",
        )
        dataset_params.add_argument(
            "--current_episode_train", type=int, default=-1
        )
        dataset_params.add_argument(
            "--current_episode_val", type=int, default=-1
        )
        dataset_params.add_argument("--min_z", type=float, default=0.5)
        dataset_params.add_argument("--max_z", type=float, default=10.0)
        dataset_params.add_argument("--W", type=int, default=256)
        dataset_params.add_argument("--motionW", type=int, default=256)
        dataset_params.add_argument("--motionH", type=int, default=256)
        dataset_params.add_argument(
            "--images_before_reset", type=int, default=1000
        )
        dataset_params.add_argument(
            "--image_type",
            type=str,
            default="both",
            choices=(
                "both",
                "translation",
                "rotation",
                "outpaint",
                "fixedRT_baseline",
            ),
        )
        dataset_params.add_argument("--max_angle", type=int, default=45)
        dataset_params.add_argument(
            "--use_z", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_inv_z", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_encoder_features", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_rgb_features", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_alpha", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_bg_blur", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_input_img_at_static_region", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_start_img_only", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_mask_as_motion_input", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--use_hint_as_motion_input", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--detach_bg", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--detach_fluid_bgalpha", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--detach_rock_fluidalpha", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--normalize_image", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--normalize_pred_motion", action="store_true", default=False
        )
        dataset_params.add_argument(
            "--normalize_pred_motion2", action="store_true", default=False
        )
        dataset_params.add_argument("--flow_raw_W", type=int, default=1920)
        dataset_params.add_argument("--flow_raw_H", type=int, default=1024)
        dataset_params.add_argument("--flow_input_W", type=int, default=480)
        dataset_params.add_argument("--flow_input_H", type=int, default=256)

    def add_train_parameters(self):
        training = self.parser.add_argument_group("training")
        training.add_argument("--num_workers", type=int, default=0)
        training.add_argument("--start-epoch", type=int, default=0)
        training.add_argument("--num-accumulations", type=int, default=1)
        training.add_argument("--lr", type=float, default=1e-3)
        training.add_argument("--lr_d", type=float, default=1e-3 * 2)
        training.add_argument("--lr_g", type=float, default=1e-3 / 2)
        training.add_argument("--alpha_lr_d", type=float, default=1e-3 * 2)
        training.add_argument("--alpha_lr_g", type=float, default=1e-3 / 2)
        training.add_argument("--momentum", type=float, default=0.9)
        training.add_argument("--beta1", type=float, default=0)
        training.add_argument("--beta2", type=float, default=0.9)
        training.add_argument("--gamma_smoothl1", type=float, default=0.1)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--init", type=str, default="")

        training.add_argument(
            "--use_fluid_img_at_static_region", action="store_true", default=False
        )
        training.add_argument(
            "--use_start_img_at_static_region", action="store_true", default=False
        )
        training.add_argument(
            "--use_multi_hypothesis", action="store_true", default=False
        )
        training.add_argument("--num_hypothesis", type=int, default=1)
        training.add_argument("--z_dim", type=int, default=128)
        training.add_argument(
            "--netD", type=str, default="multiscale", help="(multiscale)"
        )
        training.add_argument(
            "--niter",
            type=int,
            default=100,
            help="# of iter at starting learning rate. This is NOT the total #epochs."
            + " Total #epochs is niter + niter_decay",
        )
        training.add_argument(
            "--niter_decay",
            type=int,
            default=10,
            help="# of iter at starting learning rate. This is NOT the total #epochs."
            + " Totla #epochs is niter + niter_decay",
        )

        training.add_argument(
            "--losses", type=str, nargs="+", default=['1.0_l1','10.0_content']
        )
        training.add_argument(
            "--FGMThreshold", type=float, default=0.01
        )
        training.add_argument(
            "--use_MotionMaskThreshold", action="store_true", default=False
        )
        training.add_argument(
            "--MotionMaskThreshold", type=float, default=0.2161635
        )
        training.add_argument(
            "--MVloss", type=float, default=0.0
        )
        training.add_argument(
            "--StaticRegionMVloss", type=float, default=0.0
        )
        training.add_argument(
            "--MovingRegionMVloss", type=float, default=0.0
        )
        training.add_argument(
            "--StaticRegionInputImageSupervision", type=float, default=0.0
        )
        training.add_argument(
            "--balanced_weight", type=int, default=1
        )
        training.add_argument(
            "--MVlossPerceptual", type=float, default=1.0
        )
        training.add_argument(
            "--MVlossL1", type=float, default=1.0
        )
        training.add_argument(
            "--MVlossLowres", type=float, default=1.0
        )
        training.add_argument('--use_gt_mean_img', action="store_true", default=False)
        training.add_argument('--use_input_img', action="store_true", default=False)

        training.add_argument(
            "--AKLloss", type=float, default=0.0
        )
        training.add_argument(
            "--fluidimgloss", type=float, default=0.0
        )
        training.add_argument(
            "--Inpaintloss", type=float, default=0.0
        )
        training.add_argument(
            "--sigma_bias", type=float, default=0.0
        )
        training.add_argument(
            "--I0loss", type=float, default=0.0
        )
        training.add_argument(
            "--INloss", type=float, default=0.0
        )
        training.add_argument(
            "--ADCloss", type=float, default=0.0
        )
        training.add_argument(
            "--MRADCloss", type=float, default=0.0
        )
        training.add_argument(
            "--RockRegionloss", type=float, default=0.0
        )
        training.add_argument(
            "--RockRegionlossDecay", type=float, default=0.0
        )
        training.add_argument(
            "--RockRegionlosstarget", type=float, default=0.25
        )
        training.add_argument(
            "--FluidRegionloss", type=float, default=0.0
        )
        training.add_argument(
            "--TVloss", type=float, default=0.0
        )
        training.add_argument(
            "--ATVloss", type=float, default=0.0
        )
        training.add_argument(
            "--AlphaMSEloss", type=float, default=0.0
        )
        training.add_argument(
            "--AlphaL1loss", type=float, default=0.0
        )
        training.add_argument(
            "--AlphaWeightDecay", type=float, default=0.0
        )
        training.add_argument(
            "--motion_losses", type=str, nargs="+", default=['1.0_l1']
        )
        training.add_argument(
            "--discriminator_losses",
            type=str,
            default="pix2pixHD",
            help="(|pix2pixHD|progressive)",
        )
        training.add_argument(
            "--lambda_feat",
            type=float,
            default=10.0,
            help="weight for feature matching loss",
        )
        training.add_argument(
            "--gan_mode", type=str, default="hinge", help="(ls|original|hinge)"
        )

        training.add_argument(
            "--load_old_model", action="store_true", default=False
        )
        training.add_argument("--old_model", type=str, default="")
        training.add_argument("--old_depth_model", type=str, default="")

        training.add_argument(
            "--no_ganFeat_loss",
            action="store_true",
            help="if specified, do *not* use discriminator feature matching loss",
        )
        training.add_argument(
            "--no_vgg_loss",
            action="store_true",
            help="if specified, do *not* use VGG feature matching loss",
        )
        training.add_argument("--resume", action="store_true", default=False)

        training.add_argument(
            "--log-dir",
            type=str,
            default="logging/viewsynthesis3d/%s/",
        )

        training.add_argument("--batch-size", type=int, default=16)
        training.add_argument("--continue_epoch", type=int, default=0)
        training.add_argument("--max_epoch", type=int, default=500)
        training.add_argument("--folder_to_save", type=str, default="outpaint")
        training.add_argument(
            "--model_epoch_path",
            type=str,
            #default="/%s/%s/models/lr%0.5f_bs%d_model%s_spl%s/noise%s_bn%s_ref%s_d%s_"
            #+ "camxys%s/_init%s_data%s_seed%d/_multi%s_losses%s_i%s_%s_vol_gan%s/",
            default="/%s/%s/models/lr%0.5f_bs%d_model%s_spl%s_ref%s_seed%d",
        )
        training.add_argument(
            "--run-dir",
            type=str,
            default="/%s/%s/runs/lr%0.5f_bs%d_model%s_spl%s/noise%s_bn%s_ref%s_d%s_"
            + "camxys%s/_init%s_data%s_seed%d/_multi%s_losses%s_i%s_%s_vol_gan%s/",
        )
        training.add_argument("--suffix", type=str, default="")
        training.add_argument(
            "--render_ids", type=int, nargs="+", default=[0, 1]
        )
        training.add_argument("--gpu_ids", type=str, default="0")
        training.add_argument("--use_completed_depth", action="store_true", default=False)
        training.add_argument("--pconv", type=str, default="pconv")
        training.add_argument("--residual_inpainting", type=str, default="None")
        training.add_argument("--projected_loss", type=str, default="None")
        training.add_argument("--train_Z", action="store_true", default=False)
        training.add_argument("--bn_noise_misc", action="store_true", default=False)
        training.add_argument("--Z_model", type=str, default="encoder")
        training.add_argument('--resize_or_crop', type=str, default='crop_and_resize',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        training.add_argument('--no_flip', action='store_true',
                             help='if specified, do not flip the images for data argumentation')
        training.add_argument('--use_color_jitter', action='store_true')
        training.add_argument('--use_online_hint', action='store_true')
        training.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        training.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        training.add_argument('--use_sgd', action="store_true", default=False)
        training.add_argument('--load_decoder', action="store_true", default=False)
        training.add_argument('--pretrained_decoder', type=str, default='')
        training.add_argument('--load_encoder', action="store_true", default=False)
        training.add_argument('--use_linear_in_softsplat', action="store_true", default=False)
        training.add_argument('--max_video_length', type=int, default=80)
        training.add_argument('--min_video_length', type=int, default=0)
        training.add_argument('--augmented_edge_mask', action="store_true", default=False)
        training.add_argument('--augmented_edge_mask_rate', type=float, default=0.2)
        training.add_argument('--augmented_edge_mask_length', type=int, default=7)
        training.add_argument('--random_ff_mask', action="store_true", default=False)
        training.add_argument('--random_ff_mask_rate', type=float, default=0.5)
        training.add_argument('--random_ff_mask_mv', type=float, default=5)
        training.add_argument('--random_ff_mask_ma', type=float, default=4.0)
        training.add_argument('--random_ff_mask_ml', type=float, default=40)
        training.add_argument('--random_ff_mask_mbw', type=float, default=10)
        training.add_argument('--I0_only', action="store_true", default=False)
        training.add_argument('--use_perlin2d_noise', action="store_true", default=False)
        training.add_argument('--noise_blending', type=str, default='linear_before_decoder')
        training.add_argument('--noise_generator', type=str, default='encoder')
        training.add_argument('--noise_channel', type=int, default=64)
        training.add_argument('--beta_flow', type=float, default=1.0)
        training.add_argument('--loss_style', type=float, default=0.1)
        training.add_argument('--use_3d_splatter', action="store_true", default=False)
        training.add_argument('--no_clamp_Z', action="store_true", default=False)
        training.add_argument('--use_softmax_splatter', action="store_true", default=False)
        training.add_argument('--use_softmax_splatter_v1', action="store_true", default=False)
        training.add_argument('--use_softmax_splatter_v2', action="store_true", default=False)
        training.add_argument('--use_softmax_splatter_v3', action="store_true", default=False)
        training.add_argument('--use_mesh_splatter', action="store_true", default=False)
        training.add_argument('--use_single_scene', action="store_true", default=False)
        training.add_argument('--scene_id', type=str, default='00018')
        training.add_argument('--alpha_no_zero', action="store_true", default=False)
        training.add_argument('--clamp_alpha',  type=float, default=0.0)
        training.add_argument('--use_local_TV', action="store_true", default=False)
        training.add_argument('--use_bilateral_depth', action="store_true", default=False)
        training.add_argument('--ksize', type=float, default=15)
        training.add_argument('--sigmaColor', type=float, default=5)
        training.add_argument('--sigmaSpace', type=float, default=5)
        training.add_argument('--use_tanh', action="store_true", default=False)
        training.add_argument('--normalized_speed', action="store_true", default=False)
        training.add_argument('--use_reweight_mask_loss', action="store_true", default=False)
        training.add_argument('--use_reweight_mask_loss2', action="store_true", default=False)
        training.add_argument('--use_static_moving_loss', action="store_true", default=False)
        training.add_argument('--static_moving_weight', type=float, default=0.3)
        training.add_argument('--reweight_pos', type=float, default=1)
        training.add_argument('--reweight_neg', type=float, default=1)
        training.add_argument('--use_pconv_mask', action="store_true", default=False)
        training.add_argument('--use_bg_alpha', action="store_true", default=False)
        training.add_argument('--use_motion_speed_alpha', action="store_true", default=False)
        training.add_argument('--use_slower_motion', action="store_true", default=False)
        training.add_argument('--motion_speed_power', type=int, default=1)
        training.add_argument('--bg_alpha_power', type=int, default=1)
        training.add_argument('--style_loss_layer', type=int, default=1)
        training.add_argument('--out_channel', type=int, default=65)
        training.add_argument('--div_flow', type=float, default=1.0)
        training.add_argument('--use_vgg_style', action="store_true", default=False)
        training.add_argument('--train_alpha_mode', type=str, default="img")
        training.add_argument('--load_motion_regressor', action="store_true", default=False)
        training.add_argument('--motion_regressor_model', type=str, default='')
        training.add_argument('--load_bg_model', action="store_true", default=False)
        training.add_argument('--bg_model', type=str, default='')
        training.add_argument('--addtional_decoder_input', type=int, default=0)
        training.add_argument('--addtional_decoder_output', type=int, default=0)



    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
            }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)


def get_timestamp():
    ts = time.time()
    #st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    #st = "2020-08-19"
    st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())[:7]
    return st


def get_log_path(timestamp, opts):
    # logdir logging/viewsynthesis3d
    # logpath
    inputs = "z" + str(opts.use_z) + "_alpha" + str(opts.use_alpha)
    return (
        opts.log_dir % (opts.dataset)
        + "/%s/"
        + opts.run_dir
        % (
            timestamp,
            opts.folder_to_save,
            opts.lr,
            opts.batch_size,
            opts.model_type,
            opts.splatter,
            opts.noise,
            opts.norm_G,
            opts.refine_model_type,
            opts.depth_predictor_type,
            (str(opts.use_camera) + "|" + str(opts.use_xys)),
            opts.init,
            opts.image_type,
            opts.seed,
            str(opts.use_multi_hypothesis),
            "".join(opts.losses).replace("_", "|"),
            inputs,
            opts.suffix,
            opts.discriminator_losses,
        )
    )

def get_model_path(timestamp, opts):
    inputs = ""#"z" + str(opts.use_z) + "_alpha" + str(opts.use_alpha)# zFalse_alphaFalse
    model_path = opts.log_dir % (opts.dataset) + opts.model_epoch_path % (
        timestamp,
        opts.folder_to_save,
        opts.lr,
        opts.batch_size,
        opts.model_type,
        opts.splatter,
        opts.refine_model_type,
        opts.seed,
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path + "/model_epoch.pth"
