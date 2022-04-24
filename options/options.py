def get_model(opt):
    print("Loading model %s ... ")
    if opt.model_type == "zbuffer_pts":
        from models.z_buffermodel import ZbufferModelPts

        model = ZbufferModelPts(opt)
    ### Baseline ###
    elif opt.model_type == "softmax_splating":
        from models.animating_softmax_splating import AnimatingSoftmaxSplating

        model = AnimatingSoftmaxSplating(opt)
    elif opt.model_type == "softmax_splating_Zmax":
        from models.animating_softmax_splating_Zmax import AnimatingSoftmaxSplating

        model = AnimatingSoftmaxSplating(opt)
    ### CLAW ###
    elif opt.model_type == "bg":
        from models.animating_softmax_splating_2layers_alpha_seperate import BackgroundNetwork

        model = BackgroundNetwork(opt)
    elif opt.model_type == "alpha":
        from models.animating_softmax_splating_2layers_alpha_seperate import AlphaRegressor

        model = AlphaRegressor(opt)
    elif opt.model_type == "softmax_splating_2layers_alpha_seperate":
        from models.animating_softmax_splating_2layers_alpha_seperate import AnimatingSoftmaxSplatingJoint

        model = AnimatingSoftmaxSplatingJoint(opt)
    elif opt.model_type == "softmax_splating_2layers_alpha":
        from models.animating_softmax_splating_2layers_alpha import AnimatingSoftmaxSplatingAlphaEncoder

        model = AnimatingSoftmaxSplatingAlphaEncoder(opt)
    ### Motion ###
    elif opt.model_type == "unet_motion":
        from models.unet_motion import UnetMotion

        model = UnetMotion(opt)
    elif opt.model_type == "SPADE_unet_mask_motion":
        from models.unet_motion import SPADEUnetMaskMotion

        model = SPADEUnetMaskMotion(opt)
    elif opt.model_type == "softmax_splating_joint":
        from models.animating_softmax_splating_joint import AnimatingSoftmaxSplating

        model = AnimatingSoftmaxSplating(opt)
    return model

def get_dataset(opt):

    print("Loading dataset %s ..." % opt.dataset)

    if  opt.dataset == "realestate":
        opt.min_z = 1.0
        opt.max_z = 100.0
        opt.train_data_path = (
            "data/realestate10K/RealEstate10K",
            "data/files/RealEstate10K"
        )
        from data.realestate10k import RealEstate10K

        return RealEstate10K

    elif opt.dataset == "eulerian_data":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/files/eulerian_data"
        )
        from data.eulerian_data import Liquid

        return Liquid

    elif opt.dataset == "eulerian_data_hint":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/files/eulerian_data"
        )
        from data.eulerian_data_hint import Liquid

        return Liquid

    elif opt.dataset == "eulerian_data_bg":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/files/eulerian_data"
        )
        from data.eulerian_data_bg import Liquid

        return Liquid

    elif opt.dataset == "eulerian_data_balanced1_mask":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/shallow_water_all/",
        )
        opt.rock_label_data_path = "data/eulerian_data/fluid_region_rock_labels/all"
        from data.eulerian_data_balanced1_mask import Liquid

        return Liquid

    elif opt.dataset == "eulerian_data_balanced_mask_hint":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/shallow_water_all/",
        )
        opt.align_data_path = "data/eulerian_data/gtmotion_warp_outputs/align_max_frame_005_bin_max150.json"
        opt.rock_label_data_path = "data/eulerian_data/fluid_region_rock_labels/all"
        from data.eulerian_data_balanced_mask_hint import Liquid

        return Liquid
    elif opt.dataset == "eulerian_data_mask_hint":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/shallow_water_all/",
        )
        opt.rock_label_data_path = "data/eulerian_data/fluid_region_rock_labels/all"
        from data.eulerian_data_mask_hint import Liquid

        return Liquid

    elif opt.dataset == "eulerian_data_balanced1_align_mask":
        opt.train_data_path = (
            "data/eulerian_data/",
            "data/shallow_water_all/",
        )
        opt.rock_label_data_path = "data/eulerian_data/fluid_region_rock_labels/all"
        opt.align_data_path = "data/eulerian_data/align_max_frame_005_bin_max150.json"
        from data.eulerian_data_balanced1_align_mask import Liquid

        return Liquid

    print("No matching dataset {}, use HabitatImageGenerator instead.".format(opt.dataset))
    from data.habitat_data import HabitatImageGenerator as Dataset

    return Dataset
