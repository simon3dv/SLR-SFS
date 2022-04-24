import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.synthesis import SynthesisLoss
from models.networks.architectures import Unet
from models.networks.utilities import get_decoder, get_encoder
#from models.projection.animating_z_buffer_manipulator import AnimatingPtsManipulator
from models.projection.euler_integration_manipulator import EulerIntegration, euler_integration
#from pytorch3d.structures.meshes import Meshes
#from pytorch3d.renderer.mesh.textures import Textures
#from pytorch3d.renderer.mesh import rasterize_meshes
#from pytorch3d.renderer.cameras import look_at_view_transform  # , FoVPerspectiveCameras
#from utils.cameras import FoVPerspectiveCameras, PerspectiveCameras
#from pytorch3d.renderer import (
#       RasterizationSettings, BlendParams,
#       MeshRenderer, MeshRasterizer
#)
import time
from utils.utils import AverageMeter
import numpy as np
import copy
from models.losses.synthesis import PerceptualLoss
import cv2
import time
from models import softsplat

from models.networks.architectures import ResNetEncoder
from models.networks.architectures import ResNetDecoder
from models.networks.networks import define_G
from utils.utils import AverageMeter
from models.losses.synthesis import MotionLoss
from models.networks.architectures import VGG19
#from models.unet_motion import UnetMotion

DEBUG = False
DEBUG_LOSS = False


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        return img[:, :, (self.h - self.th) // 2:(self.h + self.th) // 2,
               (self.w - self.tw) // 2:(self.w + self.tw) // 2]


def softmax_rgb_blend(
                colors, fragments, blend_params, in_channel=3, znear: float = 1.0, zfar: float = 100
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
            relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
            colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
            fragments: namedtuple with outputs of rasterization. We use properties
                    - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                      of the faces (in the packed representation) which
                      overlap each pixel in the image.
                    - dists: FloatTensor of shape (N, H, W, K) specifying
                      the 2D euclidean distance from the center of each pixel
                      to each of the top K overlapping faces.
                    - zbuf: FloatTensor of shape (N, H, W, K) specifying
                      the interpolated depth from each pixel to to each of the
                      top K overlapping faces.
            blend_params: instance of BlendParams dataclass containing properties
                    - sigma: float, parameter which controls the width of the sigmoid
                      function used to calculate the 2D distance based probability.
                      Sigma controls the sharpness of the edges of the shape.
                    - gamma: float, parameter which controls the scaling of the
                      exponential function used to control the opacity of the color.
                    - background_color: (3) element list/tuple/torch.Tensor specifying
                      the RGB values for the background color.
            znear: float, near clipping plane in the z direction
            zfar: float, far clipping plane in the z direction

    Returns:
            RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = torch.ones((N, H, W, in_channel + 1), dtype=colors.dtype, device=colors.device)
    background = blend_params.background_color
    if not torch.is_tensor(background):
        background = torch.tensor(background, dtype=torch.float32, device=device)
    else:
        background = background.to(device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = (fragments.pix_to_face >= 0).float()

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[20]: Argument `max` expected.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors[..., :in_channel] = (weighted_colors + weighted_background) / denom
    pixel_colors[..., in_channel] = 1.0 - alpha

    return pixel_colors


class myRenderer(torch.nn.Module):
    def __init__(
                    self, opt, device="cpu", cameras=None, W=256, blend_params=None, in_channel=3
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams(
                background_color=[0.0] * in_channel)
        if "blur_radius" in opt:
            blur_radius = opt.blur_radius
        if "faces_per_pixel" in opt:
            faces_per_pixel = opt.faces_per_pixel
        blur_radius = 1e-8
        faces_per_pixel = 1

        raster_settings = RasterizationSettings(
                image_size=W,
                blur_radius=blur_radius,
                faces_per_pixel=faces_per_pixel,
        )
        self.rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.in_channel = in_channel

    def forward(self, meshes, **kwargs) -> torch.Tensor:
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        # meshes.verts_packed()
        fragments = self.rasterizer(meshes)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params, in_channel=self.in_channel)
        return images


class AnimatingSoftmaxSplating(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        W = opt.W
        self.div_flow = 20.0 if "div_flow" not in opt else opt.div_flow
        # ENCODER
        # Encode features to a given resolution
        self.encoder = get_encoder(opt)
        # POINT CLOUD TRANSFORMER
        # REGRESS 3D POINTS
        # self.pts_regressor = Unet(channels_in=3, channels_out=1, opt=opt)
        if "train_motion" in opt and opt.train_motion:
            from options.options import get_model
            opts_model = copy.deepcopy(self.opt)
            opts_model.model_type = self.opt.motion_model_type
            self.motion_regressor = get_model(opts_model)

            '''
            # self.motion_loss_function = MotionLoss(opt=opt)
 
            if 'uvm' in opt.motion_model_type:
                self.motion_regressor = UnetMotionUVM(opt)
            else:
                self.motion_regressor = UnetMotion(opt)
            '''

        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            self.splatter = self.get_splatter(
                    opt.splatter, None, opt, size=W, C=opt.ngf, points_per_pixel=opt.pp_pixel
            )
            # create xyzs(1,3,256x256) tensor for image
            xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
            ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

            xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
            ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

            xyzs = torch.cat(
                    (xs, ys, torch.ones(xs.size())), 1
            ).view(1, 3, -1)  # torch.Size([1, 4, 65536])
            self.register_buffer("xyzs", xyzs)
        self.euler_integration = EulerIntegration(self.opt)
        if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
            self.maximum_warp_norm_splater = softsplat.ModuleMaximumWarpNormsplat()
        self.softsplater = softsplat.ModuleSoftsplat('summation')

        self.projector = get_decoder(opt)

        # LOSS FUNCTION
        # Module to abstract away the loss function complexity
        self.loss_function = SynthesisLoss(opt=opt)

        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer(
                "max_z", torch.Tensor([self.opt.max_z])
        )
        self.discretized = self.register_buffer(
                "discretized_zs",
                torch.linspace(self.opt.min_z, self.opt.max_z, self.opt.voxel_size),
        )


    def depth2mesh(self, D, cam=None, max_cos=2., max_len=-1, eps=1e-6):
        bs = D.shape[0]
        W = D.shape[-1]
        device = D.device
        teps = torch.tensor(eps).to(device)
        x = torch.linspace(0, W - 1, W).long()  # / float(W - 1)
        y = torch.linspace(0, W - 1, W).long()  # / float(W - 1)
        x = x.view(1, W).repeat(W, 1)
        y = y.view(W, 1).repeat(1, W)
        # xyzs = torch.cat([((x.view(1,-1).float())-W/2),
        #                  ((y.view(1,-1).float())-W/2),
        #                  torch.ones((1,W*W))], dim=0).view(1, 3, -1).repeat([bs,1,1]).to(device)
        xyzs = torch.cat([((x.view(1, -1).float())) / (W - 1) * 2 - 1,
                          (((y.view(1, -1).float())) / (W - 1) * 2 - 1),
                          torch.ones((1, W * W))], dim=0).view(1, 3, -1).repeat([bs, 1, 1]).to(device)
        if cam is None:
            fx = fy = max(D.shape[:2])
            cx = D.shape[1] / 2.
            cy = D.shape[0] / 2.
            cam = np.array([ \
                    [fx, 0, cx], \
                    [0, fy, cy], \
                    [0, 0, 1]])
        # v = cam.inverse().bmm(xyzs)
        v = xyzs
        v = v * D.view(D.shape[0], 1, -1).repeat((1, 3, 1))  # torch.Size([bs, 3, W*W])
        if max_cos > 0:
            quad = torch.cat(( \
                    (x[:-1, :-1] + y[:-1, :-1] * W).view(1, -1), \
                    (x[1:, :-1] + y[1:, :-1] * W).view(1, -1), \
                    (x[1:, 1:] + y[1:, 1:] * W).view(1, -1), \
                    (x[:-1, 1:] + y[:-1, 1:] * W).view(1, -1)), dim=0).to(device)  # torch.Size([bs, 4, F=W-1*W-1])
            roll = torch.tensor([1, 2, 3, 0])
            e = (v[:, :, quad[roll, :]] - v[:, :, quad])  # torch.Size([bs,3,4,F])
            d = v[:, :, quad[2:, :]] - v[:, :, quad[:2, :]]  # torch.Size(bs, 3, 2, F])
            d = torch.cat((d, -d), 2)  # torch.Size(bs, 3, 4, W*W])

            n = torch.sqrt((torch.cat(((e * e).sum(1), (d * d).sum(1)), 1)))  # torch.Size(bs, 8, F])
            corner = -(e * e[:, :, roll, :]).sum(1) / torch.max(n[:, :4, :] * n[:, roll, :],
                                                                teps)  # torch.Size(bs, 4, F])
            split = torch.cat(( \
                    (e * d).sum(1) / torch.max(n[:, :4, :] * n[:, 4:, :], teps), \
                    (e * -d[:, :, roll, :]).sum(1) / torch.max(n[:, :4, :] * n[:, 4 + roll, :], teps)),
                    1)  # torch.Size(bs, 8, F]), cos(e_i, d_i), cos(e_i, -d_i+1), i=0,1,2,3

            tri_cos = torch.cat(( \
                    split[:, :4, :].unsqueeze(-1), \
                    corner.unsqueeze(-1), \
                    split[:, 4 + roll, :].unsqueeze(-1)), -1).max(-1)[
                    0]  # torch.Size(bs, 4, F]), # max(cos(ei,di), cos(ei,ei+1), cos(ei,di+1)), i=0,1,2,3
            tri_len = torch.cat(( \
                    n[:, :4, :].unsqueeze(-1), \
                    n[:, 4:, :].unsqueeze(-1), \
                    n[:, roll].unsqueeze(-1)), -1).max(-1)[
                    0]  # torch.Size(bs, 4, F]), # max(len(ei), len(di), len(ei+1)), i=0,1,2,3
            v_valid = (v[:, 2, :] > eps)  # (bs, F)

            quad_valid = v_valid[:, quad].sum(1)  # torch.Size(bs, F]),01234
            tri = None


            for b in range(bs):
                quad_type1 = torch.nonzero(quad_valid[b] == 3).view(-1)
                quad_type2 = torch.nonzero(
                        (quad_valid[b] == 4) & (tri_cos[b, ::2, :].max(0)[0] <= tri_cos[b, 1::2, :].max(0)[0])).view(-1)
                quad_type3 = torch.nonzero(
                        (quad_valid[b] == 4) & (tri_cos[b, ::2, :].max(0)[0] > tri_cos[b, 1::2, :].max(0)[0])).view(-1)

                # quad_type1
                J = torch.nonzero(v_valid[b, quad[:, quad_type1]] == 0).view(-1)
                I = quad_type1
                cond = (tri_cos[b, (J + 1) % 4, I] < max_cos) & (
                                (tri_len[b, (J + 1) % 4, I] < max_len) | (max_len <= 0))
                tri1 = torch.cat((quad[(J + 1) % 4, I].unsqueeze(0), quad[(J + 2) % 4, I].unsqueeze(0),
                                  quad[(J + 3) % 4, I].unsqueeze(0)), 0).transpose(0, 1)

                # tri1 = np.array([quad[i,[(j+1)%4,(j+2)%4,(j+3)%4]]
                # for i,j in zip(quad_type1, \
                # np.where(v_valid[quad[quad_type1,:]] == 0)[1]) \
                # if tri_cos[i,(j+1)%4] < max_cos and \
                # (max_len <= 0 or tri_len[i,(j+1)%4] < max_len)]).reshape((-1,3))

                # quad_type2
                cond1 = torch.nonzero((tri_cos[b, 0, :] < max_cos) & (max_len <= 0 or tri_len[b, 0, :] < max_len)).view(
                        -1)
                cond2 = torch.nonzero((tri_cos[b, 2, :] < max_cos) & (max_len <= 0 or tri_len[b, 2, :] < max_len)).view(
                        -1)

                I1, J1 = torch.meshgrid(cond1, torch.tensor([0, 1, 2]).to(device))
                I2, J2 = torch.meshgrid(cond2, torch.tensor([2, 3, 0]).to(device))
                tri2 = torch.cat(
                        (quad[J1, I1],
                         quad[J2, I2]), dim=0)

                # tri2 = np.array( \
                # [quad[i, [0, 1, 2]] for i in quad_type2 \
                # if tri_cos[i, 0] < max_cos and \
                # (max_len <= 0 or tri_len[i, 0] < max_len)] + \
                # [quad[i, [2, 3, 0]] for i in quad_type2 \
                # if tri_cos[i, 2] < max_cos and \
                # (max_len <= 0 or tri_len[i, 2] < max_len)]).reshape((-1, 3))

                cond3 = torch.nonzero((tri_cos[b, 1, :] < max_cos) & (max_len <= 0 or tri_len[b, 1, :] < max_len)).view(
                        -1)
                cond4 = torch.nonzero((tri_cos[b, 3, :] < max_cos) & (max_len <= 0 or tri_len[b, 3, :] < max_len)).view(
                        -1)
                I3, J3 = torch.meshgrid(cond3, torch.tensor([1, 2, 3]).to(device))
                I4, J4 = torch.meshgrid(cond4, torch.tensor([3, 0, 1]).to(device))
                tri3 = torch.cat(
                        (quad[J3, I3],
                         quad[J4, I4]), dim=0)
                # tri_b = torch.cat((tri1, tri2, tri3), 0)
                tri_b = torch.cat((tri1, tri2), 0)
                # quad_type3
                # tri3 = np.array( \
                #       [quad[i, [1, 2, 3]] for i in quad_type3 \
                #        if tri_cos[i, 1] < max_cos and \
                #        (max_len <= 0 or tri_len[i, 1] < max_len)] + \
                #       [quad[i, [3, 0, 1]] for i in quad_type3 \
                #        if tri_cos[i, 3] < max_cos and \
                #        (max_len <= 0 or tri_len[i, 3] < max_len)]).reshape((-1, 3))

                if type(tri) == type(None):
                    tri = tri_b.unsqueeze(0)
                else:
                    tri = torch.cat((tri, tri_b.unsqueeze(0)), 0)
        else:
            tri = torch.zeros((bs, 0, 3)).float().to(device)
        '''
        mean_depth = D.mean()
        for b in range(bs):
                depth_edge_valid = torch.zeros(tri.shape[1]).float().cuda()
                for edge_id in [[0, 1], [1, 2], [2, 0]]:
                        edges = tri[b, :, edge_id]  # n_faces, 2
                        depth_edge = v[b, -1, edges]
                        depth_edge_valid += ((depth_edge[:, 0] - depth_edge[:, 1]).abs() < config['depth_threshold']).float()
                depth_edge_cond = torch.nonzero(
                        (depth_edge_valid >= 3).float() + (depth_edge[:, 0] >= mean_depth).float()).squeeze()
                if b == 0:
                        new_tri = tri[0:1, depth_edge_cond, :]
                else:
                        new_tri = torch.cat([new_tri, tri[b:b + 1, depth_edge_cond, :]], 0)
        '''
        return v.transpose(1, 2), tri

    def get_splatter(
                    self, name, depth_values, opt=None, size=256, C=64, points_per_pixel=8
    ):
        if name == "xyblending":
            from models.layers.z_buffer_layers import RasterizePointsXYsBlending

            return RasterizePointsXYsBlending(
                    C,
                    learn_feature=opt.learn_default_feature,  # False
                    radius=opt.radius,
                    size=size,
                    points_per_pixel=points_per_pixel,
                    opts=opt,
            )
        elif name == "xyblending":
            from models.layers.z_buffer_layers import RasterizePointsXYAlphasBlending

            return RasterizePointsXYAlphasBlending(
                    C,
                    learn_feature=opt.learn_default_feature,  # False
                    radius=opt.radius,
                    size=size,
                    points_per_pixel=points_per_pixel,
                    opts=opt,
            )
        else:
            raise NotImplementedError()

    def random_ff_mask(self):
        """Generate a random free form mask with configuration.
        Args:
                config: Config should have configuration including IMG_SHAPES,
                        VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
                tuple: (top, left, height, width)
        """

        h = self.opt.W
        w = self.opt.W
        mask = np.zeros((h, w))
        num_v = 12 + np.random.randint(
                self.opt.random_ff_mask_mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(self.opt.random_ff_mask_ma)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(self.opt.random_ff_mask_ml)
                brush_w = 10 + np.random.randint(self.opt.random_ff_mask_mbw)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return mask.reshape((1, 1,) + mask.shape).astype(np.float32)

    def forward(self, batch):
        """ Forward pass of a view synthesis model with a voxel latent field.
        """
        # Input values
        start_img = batch["images"][0]  # B, 3, W, W
        middle_img = batch["images"][1]
        end_img = batch["images"][2]

        bs = start_img.shape[0]

        if "depths" in batch.keys():
            depth_img = batch["depths"][0]
        else:
            depth_img = torch.ones(start_img[:, 0:1, :, :].shape)  # B, 1, W, W
        # flow_f = batch["flow"][0] # B, 2, W, W
        # flow_p = batch["flow"][1]

        if isinstance(batch["index"], list):
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]
        elif len(batch["index"].shape) > 1:
            start_index = batch["index"][:, 0]
            middle_index = batch["index"][:, 1]
            end_index = batch["index"][:, 2]
        else:
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]

        if torch.cuda.is_available():
            start_img = start_img.cuda()
            middle_img = middle_img.cuda()
            end_img = end_img.cuda()
            if "depths" in batch.keys():
                depth_img = depth_img.cuda()
        ngf = self.opt.ngf
        if "skip_warp" in self.opt.refine_model_type:
            start_fs, Z_f, start_fs_multi_res = self.encoder(start_img)
            end_fs, Z_p, end_fs_multi_res = self.encoder(end_img)
        else:
            start_fs, Z_f = self.encoder(start_img)
            end_fs, Z_p = self.encoder(end_img)
        if "Z_model" in self.opt and "relu" in self.opt.Z_model:
            Z_f = F.relu(Z_f)
            Z_p = F.relu(Z_p)

        # Regressed points

        if "depth" in batch.keys():
            if not (self.opt.use_gt_depth):
                if not ('use_inverse_depth' in self.opt) or not (self.opt.use_inverse_depth):
                    regressed_pts = (
                                    nn.Sigmoid()(self.pts_regressor(start_img))
                                    * (self.opt.max_z - self.opt.min_z)
                                    + self.opt.min_z
                    )
                else:  # Kitti
                    # Use the inverse for datasets with landscapes, where there
                    # is a long tail on the depth distribution
                    depth = self.pts_regressor(start_img)  # torch.Size([1, 1, 256, 256])
                    regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 1. / 256)
            else:
                regressed_pts = depth_img
        else:
            regressed_pts = depth_img
        regressed_pts = regressed_pts.view(bs, 1, -1).cuda()

        # Regreesed Motion
        if "train_motion" in self.opt and self.opt.train_motion:
            flow_gt = batch["motions"]
            batch4motion = {
                    "images": [start_img],
                    "motions": batch["motions"],
            }
            if "hints" in batch.keys():
                batch4motion['hints'] = batch['hints']
            motion_loss, outputs_motion = self.motion_regressor.forward(
                    batch4motion)
            flow_uvm = outputs_motion["PredMotion"]
            if flow_uvm.shape[1] == 2:
                flow = flow_uvm.view(bs, 2, self.opt.W, self.opt.W)
                use_uvm = False
            elif flow_uvm.shape[1] == 3:
                flow = flow_uvm[:, :2, ...] * flow_uvm[:, 2:3, ...]
                flow = flow.view(bs, 2, self.opt.W,self.opt.W)
                use_uvm = True
            else:
                print("Not Implemented")
            flow_scale = [self.opt.W / self.opt.motionW, self.opt.W / self.opt.motionH]
            flow = flow * torch.FloatTensor(flow_scale).view(1,2,1,1).to(flow.device)  # train in W256
            flow = flow.view(bs,2,-1)
        else:
            if batch["motions"].shape[1] == 2:
                flow = batch["motions"].view(bs, 2, -1).cuda()
                use_uvm = False
            elif batch["motions"].shape[1] == 3:
                use_uvm = True
                flow_uvm = batch["motions"]
                flow = batch["motions"][:, :2, ...].clone()
                flow = flow * batch["motions"][:, 2:3, ...]
                flow = flow.view(bs, 2, -1).cuda()

        if "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            K = torch.FloatTensor(
                    [
                            [1.0, 0.0, 0.0, 0.0],
                            [0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                    ],
            ).unsqueeze(0)
            offsetK_coord = torch.FloatTensor(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ).unsqueeze(0)
            K = torch.bmm(offsetK_coord, K).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            fx = torch.FloatTensor([0.5])
            fov = 360 / np.pi * torch.atan(0.5 / fx)
            RT = torch.eye(4).unsqueeze(0).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            RT = torch.bmm(K, RT)
            # Render Scene
            raster_settings = RasterizationSettings(
                    image_size=self.opt.W,
                    blur_radius=self.opt.blur_radius,
                    faces_per_pixel=self.opt.faces_per_pixel,
            )

            cameras = FoVPerspectiveCameras(device=start_img.device, R=RT[:, :3, :3],
                                            T=RT[:, :3, 3], fov=fov)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            v, tri = self.depth2mesh(regressed_pts.view(bs, self.opt.W, self.opt.W), K[:, :3, :3])

        # Euler Integration
        flow = flow.view(bs,2,self.opt.W,self.opt.W).contiguous()
        flow_f = self.euler_integration(flow, middle_index.long() - start_index.long())
        flow_p = self.euler_integration(-flow, end_index.long() + 1 - middle_index.long())
        #linear
        #flow_f = flow * (middle_index.long() - start_index.long() - 1).view(2,1,1,1)
        #flow_p = -flow * ( end_index.long() + 1 - middle_index.long() - 1).view(2,1,1,1)

        alpha = (1.0 - (middle_index.float() - start_index.float()).float() / (
                                end_index.float() - start_index.float() + 1.0).float()).view(bs, 1, 1, 1).cuda()
        # forward
        if ("train_Z" not in self.opt) or (not self.opt.train_Z):
            Z_f = start_fs.new_ones([start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3]])
        Z_f = Z_f.view(bs, 1, self.opt.W, self.opt.W)


        if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
            Z_f_max = self.maximum_warp_norm_splater(tenInput=Z_f.contiguous().detach().clone(), tenFlow=flow_f, )
            Z_f_norm = Z_f - Z_f_max
        elif "use_softmax_splatter_v1" in self.opt and self.opt.use_softmax_splatter_v1:
            Z_f_norm = Z_f
        elif "use_softmax_splatter_v3" in self.opt and self.opt.use_softmax_splatter_v3:
            Z_f_norm = nn.Sigmoid()(Z_f) * 20
        else:
            Z_f_norm = Z_f - Z_f.max()
        if "no_clamp_Z" in self.opt:
            pass
        else:
            Z_f_norm = torch.clamp(Z_f_norm, min=-20.0, max=20.0)
        tenInput_f = torch.cat([start_fs * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)  # B, 65, W, W
        # 3D Splatter

        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            forward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                              self.opt.W)  # 1 3 W W
            forward_pts3D[:, :2, :, :] += flow_f / (self.opt.W / 2)  # forward_flow: 1, 2 W W
            forward_pts3D = forward_pts3D.view(bs, 3, -1)
            forward_pts3D = forward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_f = self.splatter(forward_pts3D,
                                     tenInput_f.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            forward_v = v.clone()
            forward_v[:, :, :2] += flow_f.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            forward_textures = Textures(
                    verts_rgb=(tenInput_f).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            forward_meshes = Meshes(verts=forward_v,
                            faces=tri,
                            textures=forward_textures)
            forward_fragments = rasterizer(forward_meshes)
            gen_fs_f = forward_meshes.sample_textures(forward_fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
        else:
            gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                        tenFlow=flow_f,
                                        tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                     start_fs.shape[3])))  # B, 65, W, W
        tenNormalize = gen_fs_f[:, -1:, :, :]  # B, 1, W, W
        gen_fs_f = gen_fs_f[:, :-1, :, :]
        gen_fs = gen_fs_f

        # backward
        if ("train_Z" not in self.opt) or (not self.opt.train_Z):
            Z_p = start_fs.new_ones([end_fs.shape[0], 1, end_fs.shape[2], end_fs.shape[3]])
        Z_p = Z_p.view(bs, 1, self.opt.W, self.opt.W)
        if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
            Z_p_max = self.maximum_warp_norm_splater(tenInput=Z_p.contiguous().detach().clone(),tenFlow=flow_p,)
            Z_p_norm = Z_p - Z_p_max
        elif "use_softmax_splatter_v1" in self.opt and self.opt.use_softmax_splatter_v1:
            Z_p_norm = Z_p
        else:
            Z_p_norm = Z_p - Z_p.max()
        if "no_clamp_Z" in self.opt:
            pass
        else:
            Z_p_norm = torch.clamp(Z_p_norm, min=-20.0, max=20.0)
        tenInput_p = torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha), Z_p_norm.exp() * (1 - alpha)], 1)
        # 3D Splatter
        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            backward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                               self.opt.W)  # 1 3 W W
            backward_pts3D[:, :2, :, :] += flow_p / (self.opt.W / 2)
            backward_pts3D = backward_pts3D.view(bs, 3, -1)
            backward_pts3D = backward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_p = self.splatter(backward_pts3D,
                                     tenInput_p.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            backward_v = v.clone()
            backward_v[:, :, :2] += flow_p.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            backward_textures = Textures(
                    verts_rgb=(tenInput_p).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            backward_meshes = Meshes(verts=backward_v,
                            faces=tri,
                            textures=backward_textures)
            backward_fragments = rasterizer(backward_meshes)
            gen_fs_p = backward_meshes.sample_textures(backward_fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
        else:
            gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                        tenFlow=flow_p,
                                        tenMetric=start_fs.new_ones(
                                                (start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3])))
        tenNormalize += gen_fs_p[:, -1:, :, :]
        gen_fs_p = gen_fs_p[:, :-1, :, :]
        gen_fs += gen_fs_p

        if "random_ff_mask" in self.opt and self.opt.random_ff_mask:
            random_ff_mask_flag = False
            if np.random.rand(1) < self.opt.random_ff_mask_rate:
                random_ff_mask_flag = True
                generated_mask = torch.FloatTensor(self.random_ff_mask())
                generated_mask = 1.0 - generated_mask
                generated_mask = generated_mask.repeat(gen_fs.shape[0], 1, 1, 1)
                generated_mask = generated_mask.to(gen_fs.device)
                gen_fs = gen_fs * generated_mask


        tenNormalize = torch.clamp(tenNormalize, min=1e-8)
        gen_fs = gen_fs / tenNormalize

        if "skip_warp" in self.opt.refine_model_type:
            warp_features = []
            if  "1111" in self.opt.refine_model_type:
                down_samples = [False, False, True, True]
            elif "111" in self.opt.refine_model_type:
                down_samples = [False, True, True]
            for i in range(len(start_fs_multi_res)):
                # forward
                start_fs = start_fs_multi_res[i]
                if down_samples[i]:
                    Z_f_norm = F.interpolate(Z_f_norm,
                                             (Z_f_norm.shape[2] // 2, Z_f_norm.shape[3] // 2),
                                             mode='bilinear')  # 128, 64, 32
                    flow_f = F.interpolate(flow_f, (flow_f.shape[2] // 2, flow_f.shape[3] // 2),
                                           mode='bilinear')  # 128, 64, 32
                    flow_f = flow_f / 2
                tenInput_f = torch.cat([start_fs * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)  # B, 65, W, W
                gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                                    tenFlow=flow_f,
                                                    tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                                 start_fs.shape[3])))
                warp_fea = gen_fs_f[:, :-1, :, :].clone()
                tenNormalize = gen_fs_f[:, -1:, :, :].clone()

                # backward
                end_fs = end_fs_multi_res[i]
                if down_samples[i]:
                    Z_p_norm = F.interpolate(Z_p_norm,
                                             (Z_p_norm.shape[2] // 2, Z_p_norm.shape[3] // 2),
                                             mode='bilinear')  # 128, 64, 32
                    flow_p = F.interpolate(flow_p, (flow_p.shape[2] // 2, flow_p.shape[3] // 2),
                                           mode='bilinear')  # 128, 64, 32
                    flow_p = flow_p / 2
                tenInput_p = torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha), Z_p_norm.exp() * (1 - alpha)], 1)
                gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                            tenFlow=flow_p,
                                            tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                         start_fs.shape[3])))
                warp_fea += gen_fs_p[:, :-1, :, :].clone()
                tenNormalize += gen_fs_p[:, -1:, :, :].clone()

                tenNormalize[tenNormalize == 0.0] = 1.0
                warp_fea = warp_fea / tenNormalize
                warp_features.append(warp_fea.contiguous())

        if "multires" in self.opt.refine_model_type or "skip_warp" in self.opt.refine_model_type:
            gen_img = self.projector(gen_fs.contiguous(), warp_features)
        else:
            gen_img = self.projector(gen_fs.contiguous())  # B,3,W,W, [-1,1]

        gen_img = nn.Tanh()(gen_img)
        # And the loss
        loss = self.loss_function(gen_img, middle_img)

        if "train_motion" in self.opt and self.opt.train_motion:
            lambdas, loss_names = zip(*[l.split("_") for l in self.opt.motion_losses])
            lambdas = [float(l) for l in lambdas]
            # motion_loss = self.motion_loss_function(flow, flow_gt)
            loss["Total Loss"] += motion_loss["Total Loss"]
            for i, loss_name in enumerate(loss_names):
                loss[loss_name] = motion_loss[loss_name]# * lambdas[i]
        pred_dict = {
                #"StartImg": start_img,
                "OutputImg": middle_img,
                #"EndImg": end_img,
                "PredImg": gen_img,
                "Z_f": Z_f_norm,
                #"PconvMask": (gen_fs == 0).sum(1,True).float(),
                "GTMotion": flow.view(bs, 2, self.opt.W, self.opt.W)}

        if "train_motion" in self.opt and self.opt.train_motion:
            pred_dict["PredMotion"] = outputs_motion["PredMotion"]
            pred_dict["GTMotion"] = outputs_motion["GTMotion"]

        if isinstance(pred_dict, tuple):
            pred_dict = pred_dict[0]

        for key in pred_dict.keys():
            if isinstance(pred_dict[key], tuple):
                pred_dict[key] = pred_dict[key][0]

        return loss, pred_dict

    def forward_flow(self, batch, vis_time=False):
        # Input values
        start_fs = batch["features"][0]
        bs = start_fs[0].shape[0]
        if len(start_fs) == 3:
            start_fs, Z_f, start_fs_multi_res = start_fs
        else:
            start_fs, Z_f = start_fs

        Z_f = Z_f.view(bs, 1, self.opt.W, self.opt.W)

        if "images" in batch.keys():
            image = batch["images"][0].cuda()

        if isinstance(batch["index"], list):
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]
        elif len(batch["index"].shape) > 1:
            start_index = batch["index"][:, 0]
            middle_index = batch["index"][:, 1]
            end_index = batch["index"][:, 2]
        else:
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]


        if "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            K = torch.FloatTensor(
                    [
                            [1.0, 0.0, 0.0, 0.0],
                            [0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                    ],
            ).unsqueeze(0)
            offsetK_coord = torch.FloatTensor(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ).unsqueeze(0)
            K = torch.bmm(offsetK_coord, K).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            fx = torch.FloatTensor([0.5])
            fov = 360 / np.pi * torch.atan(0.5 / fx)
            RT = torch.eye(4).unsqueeze(0).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            RT = torch.bmm(K, RT)
            # Render Scene
            raster_settings = RasterizationSettings(
                    image_size=self.opt.W,
                    blur_radius=self.opt.blur_radius,
                    faces_per_pixel=self.opt.faces_per_pixel,
            )

            cameras = FoVPerspectiveCameras(device=start_fs.device, R=RT[:, :3, :3],
                                            T=RT[:, :3, 3], fov=fov)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            v, tri = self.depth2mesh(regressed_pts.view(bs, self.opt.W, self.opt.W), K[:, :3, :3])

        # Regreesed Motion
        if "motions" not in batch.keys():
            if "motion_model_type" in self.opt and 'resnet' in self.opt.motion_model_type:
                flow = self.MotionEncoder(image)
                flow = self.MotionDecoder(flow)
            else:
                flow = self.motion_regressor.forward_flow(image)["PredMotion"]
        elif len(batch["motions"]) == 2:
            flow_f = batch["motions"][0].cuda()
            flow_p = batch["motions"][1].cuda()
        else:
            flow = batch["motions"][0].cuda()
        # start_index, middle_index, end_index = batch['index'][0]
        forward_flow, _ = euler_integration(flow, middle_index - start_index)
        backward_flow, _ = euler_integration(-flow, end_index - middle_index + 1)
        if "use_softmax_splatter_v2" in self.opt and self.opt.use_softmax_splatter_v2:
            Z_f_max = self.maximum_warp_norm_splater(tenInput=Z_f.contiguous().detach().clone(), tenFlow=forward_flow, )
            Z_f_norm = Z_f - Z_f_max
        elif "use_softmax_splatter_v1" in self.opt and self.opt.use_softmax_splatter_v1:
            Z_f_norm = Z_f
        else:
            Z_f_norm = Z_f - Z_f.max()
        if "no_clamp_Z" in self.opt:
            pass
        else:
            Z_f_norm = torch.clamp(Z_f_norm, min=-20.0, max=20.0)
        alpha = (1.0 - (middle_index - start_index).float() / (end_index - start_index + 1).float()).view(bs, 1, 1,
                                                                                                          1).cuda()
        tenInput_f = torch.cat([start_fs * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)


        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            forward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                              self.opt.W)  # 1 3 W W
            forward_pts3D[:, :2, :, :] += forward_flow / (self.opt.W / 2)  # forward_flow: 1, 2 W W
            forward_pts3D = forward_pts3D.view(bs, 3, -1)
            forward_pts3D = forward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_f = self.splatter(forward_pts3D,
                                     tenInput_f.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            forward_v = v.clone()
            forward_v[:, :, :2] += forward_flow.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            textures = Textures(
                    verts_rgb=(tenInput_f).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            meshes = Meshes(verts=forward_v,
                            faces=tri,
                            textures=textures)
            fragments = rasterizer(meshes)
            gen_fs_f = meshes.sample_textures(fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
        else:
            gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                        tenFlow=forward_flow,
                                        tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                     start_fs.shape[3])))  # B, 65, W, W



        gen_fs = gen_fs_f[:, :-1, :, :]
        tenNormalize = gen_fs_f[:, -1:, :, :]

        # backward
        tenInput_p = torch.cat([start_fs * Z_f_norm.exp() * (1 - alpha), Z_f_norm.exp() * (1 - alpha)], 1)
        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            backward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                               self.opt.W)  # 1 3 W W
            backward_pts3D[:, :2, :, :] += backward_flow / (self.opt.W / 2)
            backward_pts3D = backward_pts3D.view(bs, 3, -1)
            backward_pts3D = backward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_p = self.splatter(backward_pts3D,
                                     tenInput_p.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            backward_v = v.clone()
            backward_v[:, :, :2] += backward_flow.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            textures = Textures(
                    verts_rgb=(tenInput_p).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            meshes = Meshes(verts=backward_v,
                            faces=tri,
                            textures=textures)
            fragments = rasterizer(meshes)
            gen_fs_p = meshes.sample_textures(fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])

        else:
            gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                        tenFlow=backward_flow,
                                        tenMetric=start_fs.new_ones(
                                                (start_fs.shape[0], 1, start_fs.shape[2], start_fs.shape[3])))
        gen_fs += gen_fs_p[:, :-1, :, :]
        tenNormalize += gen_fs_p[:, -1:, :, :]

        tenNormalize = torch.clamp(tenNormalize, min=1e-8)
        gen_fs = gen_fs / tenNormalize


        if "skip_warp" in self.opt.refine_model_type:
            warp_features = []
            if  "1111" in self.opt.refine_model_type:
                down_samples = [False, False, True, True]
            elif "111" in self.opt.refine_model_type:
                down_samples = [False, True, True]
            for i in range(len(start_fs_multi_res)):
                # forward
                start_fs = start_fs_multi_res[i]
                if down_samples[i]:
                    Z_f_norm = F.interpolate(Z_f_norm,
                                             (Z_f_norm.shape[2] // 2, Z_f_norm.shape[3] // 2),
                                             mode='bilinear')  # 128, 64, 32
                    flow_f = F.interpolate(flow_f, (flow_f.shape[2] // 2, flow_f.shape[3] // 2),
                                           mode='bilinear')  # 128, 64, 32
                    flow_f = flow_f / 2
                tenInput_f = torch.cat([start_fs * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)  # B, 65, W, W
                gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                                    tenFlow=flow_f,
                                                    tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                                 start_fs.shape[3])))
                warp_fea = gen_fs_f[:, :-1, :, :].clone()
                tenNormalize = gen_fs_f[:, -1:, :, :].clone()

                # backward
                end_fs = end_fs_multi_res[i]
                if down_samples[i]:
                    Z_p_norm = F.interpolate(Z_p_norm,
                                             (Z_p_norm.shape[2] // 2, Z_p_norm.shape[3] // 2),
                                             mode='bilinear')  # 128, 64, 32
                    flow_p = F.interpolate(flow_p, (flow_p.shape[2] // 2, flow_p.shape[3] // 2),
                                           mode='bilinear')  # 128, 64, 32
                    flow_p = flow_p / 2
                tenInput_p = torch.cat([end_fs * Z_p_norm.exp() * (1 - alpha), Z_p_norm.exp() * (1 - alpha)], 1)
                gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                            tenFlow=flow_p,
                                            tenMetric=start_fs.new_ones((start_fs.shape[0], 1, start_fs.shape[2],
                                                                         start_fs.shape[3])))
                warp_fea += gen_fs_p[:, :-1, :, :].clone()
                tenNormalize += gen_fs_p[:, -1:, :, :].clone()

                tenNormalize[tenNormalize == 0.0] = 1.0
                warp_fea = warp_fea / tenNormalize
                warp_features.append(warp_fea.contiguous())

        if  "skip_warp" in self.opt.refine_model_type:
            gen_img = self.projector(gen_fs.contiguous(), warp_features)
        else:
            gen_img = self.projector(gen_fs.contiguous())  # B,3,W,W, [-1,1]

        gen_img = nn.Tanh()(gen_img)

        pred_dict = {"PredImg": gen_img}
        pred_dict["Z_f"] = Z_f
        return pred_dict

    def warp_flow(self, batch, isval=False, vis_forward_backward=False):
        # Input values
        start_img = batch["images"][0]
        end_img = start_img  # batch["features"][1]
        bs = start_img.shape[0]

        if "depths" in batch.keys():
            depth_img = batch["depths"][0]
        else:
            depth_img = torch.ones(start_img[:, 0:1, :, :].shape)

        if isinstance(batch["index"], list):
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]
        elif len(batch["index"].shape) > 1:
            start_index = batch["index"][:, 0]
            middle_index = batch["index"][:, 1]
            end_index = batch["index"][:, 2]
        else:
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]

        # Regressed points
        if "depth" in batch.keys():
            if not (self.opt.use_gt_depth):
                if not ('use_inverse_depth' in self.opt) or not (self.opt.use_inverse_depth):
                    regressed_pts = (
                                    nn.Sigmoid()(self.pts_regressor(start_img))
                                    * (self.opt.max_z - self.opt.min_z)
                                    + self.opt.min_z
                    )
                else:  # Kitti
                    # Use the inverse for datasets with landscapes, where there
                    # is a long tail on the depth distribution
                    depth = self.pts_regressor(start_img)  # torch.Size([1, 1, 256, 256])
                    regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 1. / 256)
            else:
                regressed_pts = depth_img
        else:
            regressed_pts = depth_img
        regressed_pts = regressed_pts.cuda()

        if "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            K = torch.FloatTensor(
                    [
                            [1.0, 0.0, 0.0, 0.0],
                            [0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                    ],
            ).unsqueeze(0)
            offsetK_coord = torch.FloatTensor(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ).unsqueeze(0)
            K = torch.bmm(offsetK_coord, K).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            fx = torch.FloatTensor([0.5])
            fov = 360 / np.pi * torch.atan(0.5 / fx)
            RT = torch.eye(4).unsqueeze(0).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            RT = torch.bmm(K, RT)
            # Render Scene
            raster_settings = RasterizationSettings(
                    image_size=self.opt.W,
                    blur_radius=self.opt.blur_radius,
                    faces_per_pixel=self.opt.faces_per_pixel,
            )

            cameras = FoVPerspectiveCameras(device=start_img.device, R=RT[:, :3, :3],
                                            T=RT[:, :3, 3], fov=fov)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            v, tri = self.depth2mesh(regressed_pts.view(bs, self.opt.W, self.opt.W), K[:, :3, :3])

        if "motions" not in batch.keys():
            flow = self.motion_regressor.forward_flow(image)["PredMotion"]
        elif len(batch["motions"]) == 2:
            flow_f = batch["motions"][0].cuda()
            flow_p = batch["motions"][1].cuda()
        else:
            flow = batch["motions"][0].cuda()

        # forward
        alpha = (1.0 - (middle_index - start_index).float() / (end_index - start_index).float()).view(bs, 1, 1,
                                                                                                      1).cuda()
        Z_f = start_img.new_ones([start_img.shape[0], 1, start_img.shape[2], start_img.shape[3]])  # B, 1, W, W
        Z_f_norm = Z_f - Z_f.max()
        tenInput_f = torch.cat([start_img * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)
        forward_flow = flow_f[middle_index - start_index]  # 0->t

        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            forward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                              self.opt.W)  # 1 3 W W
            forward_pts3D[:, :2, :, :] += forward_flow / (self.opt.W / 2)  # forward_flow: 1, 2 W W
            forward_pts3D = forward_pts3D.view(bs, 3, -1)
            forward_pts3D = forward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_f = self.splatter(forward_pts3D,
                                     tenInput_f.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            forward_v = v.clone()
            forward_v[:, :, :2] += forward_flow.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            textures = Textures(
                    verts_rgb=(tenInput_f).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            meshes = Meshes(verts=forward_v,
                            faces=tri,
                            textures=textures)
            fragments = rasterizer(meshes)
            gen_fs_f = meshes.sample_textures(fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
            #texels = meshes.sample_textures(fragments)
            #blend_params = BlendParams(background_color=[0.0] * tenInput_f.shape[1])
            #gen_fs_f = softmax_rgb_blend(texels, fragments, blend_params, in_channel=tenInput_f.shape[1])[:, :, :, :-1].permute([0, 3, 1, 2])
        else:
            gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                        tenFlow=forward_flow,
                                        tenMetric=start_img.new_ones((start_img.shape[0], 1, start_img.shape[2],
                                                                     start_img.shape[3])))  # B, 65, W, W

        gen_fs = gen_fs_f[:, :-1, :, :]
        tenNormalize = gen_fs_f[:, -1:, :, :]

        # backward
        backward_flow = flow_p[end_index - middle_index]  # 0->-(N-t) 0->t-N
        tenInput_p = torch.cat([end_img * Z_f.exp() * (1 - alpha), Z_f.exp() * (1 - alpha)], 1)
        # 3D Splatter
        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            backward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                               self.opt.W)  # 1 3 W W
            backward_pts3D[:, :2, :, :] += backward_flow / (self.opt.W / 2)
            backward_pts3D = backward_pts3D.view(bs, 3, -1)
            backward_pts3D = backward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_p = self.splatter(backward_pts3D,
                                     tenInput_p.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            backward_v = v.clone()
            backward_v[:, :, :2] += backward_flow.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            textures = Textures(
                    verts_rgb=(tenInput_p).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            meshes = Meshes(verts=backward_v,
                            faces=tri,
                            textures=textures)
            fragments = rasterizer(meshes)
            gen_fs_p = meshes.sample_textures(fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
            #texels = meshes.sample_textures(fragments)
            #gen_fs_p = softmax_rgb_blend(texels, fragments, blend_params, in_channel=tenInput_p.shape[1])[:, :, :, :-1].permute([0, 3, 1, 2])

        else:
            gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                        tenFlow=backward_flow,
                                        tenMetric=end_img.new_ones(
                                                (end_img.shape[0], 1, end_img.shape[2], end_img.shape[3])))

        gen_fs += gen_fs_p[:, :-1, :, :]
        tenNormalize += gen_fs_p[:, -1:, :, :]

        #tenNormalize[tenNormalize == 0.0] = 1.0
        tenNormalize = torch.clamp(tenNormalize, min=1e-8)
        gen_fs = gen_fs / tenNormalize
        gen_img = gen_fs
        pred_dict = {"PredImg": gen_img}
        if vis_forward_backward:
            alpha = 1
            tenInput_f = torch.cat([start_img * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)

            forward_flow = flow_f[middle_index - start_index]
            gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                        tenFlow=forward_flow.view(bs, -1, self.opt.W, self.opt.W),
                                        tenMetric=start_img.new_ones((bs, 1, self.opt.W, self.opt.W)))

            gen_fs = gen_fs_f[:, :-1, :, :]
            tenNormalize = gen_fs_f[:, -1:, :, :]
            #tenNormalize[tenNormalize == 0.0] = 1.0
            tenNormalize = torch.clamp(tenNormalize, min=1e-8)
            gen_fs = gen_fs / tenNormalize
            gen_img_forward = gen_fs
            pred_dict["ForawardImg"] = gen_img_forward

            alpha = 0
            tenInput_p = torch.cat([end_img * Z_f.exp() * (1 - alpha), Z_f.exp() * (1 - alpha)], 1)
            backward_flow = flow_p[end_index - middle_index]
            gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                        tenFlow=backward_flow.view(bs, -1, self.opt.W, self.opt.W),
                                        tenMetric=start_img.new_ones((bs, 1, self.opt.W, self.opt.W)))
            gen_fs = gen_fs_p[:, :-1, :, :]
            tenNormalize = gen_fs_p[:, -1:, :, :]
            #tenNormalize[tenNormalize == 0.0] = 1.0
            tenNormalize = torch.clamp(tenNormalize, min=1e-8)
            gen_fs = gen_fs / tenNormalize
            gen_img_backward = gen_fs

            pred_dict["BackwardImg"] = gen_img_backward

        return pred_dict


    def warp_flow_sparse2densemotion(self, batch, isval=False, vis_forward_backward=False):
        # Input values
        start_img = batch["images"][0]
        end_img = start_img  # batch["features"][1]
        bs = start_img.shape[0]

        if "depths" in batch.keys():
            depth_img = batch["depths"][0]
        else:
            depth_img = torch.ones(start_img[:, 0:1, :, :].shape)

        if isinstance(batch["index"], list):
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]
        elif len(batch["index"].shape) > 1:
            start_index = batch["index"][:, 0]
            middle_index = batch["index"][:, 1]
            end_index = batch["index"][:, 2]
        else:
            start_index = batch["index"][0]
            middle_index = batch["index"][1]
            end_index = batch["index"][2]

        # Regressed points
        if "depth" in batch.keys():
            if not (self.opt.use_gt_depth):
                if not ('use_inverse_depth' in self.opt) or not (self.opt.use_inverse_depth):
                    regressed_pts = (
                                    nn.Sigmoid()(self.pts_regressor(start_img))
                                    * (self.opt.max_z - self.opt.min_z)
                                    + self.opt.min_z
                    )
                else:  # Kitti
                    # Use the inverse for datasets with landscapes, where there
                    # is a long tail on the depth distribution
                    depth = self.pts_regressor(start_img)  # torch.Size([1, 1, 256, 256])
                    regressed_pts = 1. / (nn.Sigmoid()(depth) * 10 + 1. / 256)
            else:
                regressed_pts = depth_img
        else:
            regressed_pts = depth_img
        regressed_pts = regressed_pts.cuda()

        if "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            K = torch.FloatTensor(
                    [
                            [1.0, 0.0, 0.0, 0.0],
                            [0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                    ],
            ).unsqueeze(0)
            offsetK_coord = torch.FloatTensor(
                    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            ).unsqueeze(0)
            K = torch.bmm(offsetK_coord, K).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            fx = torch.FloatTensor([0.5])
            fov = 360 / np.pi * torch.atan(0.5 / fx)
            RT = torch.eye(4).unsqueeze(0).view(1, 4, 4).repeat(bs, 1, 1).cuda()
            RT = torch.bmm(K, RT)
            # Render Scene
            raster_settings = RasterizationSettings(
                    image_size=self.opt.W,
                    blur_radius=self.opt.blur_radius,
                    faces_per_pixel=self.opt.faces_per_pixel,
            )

            cameras = FoVPerspectiveCameras(device=start_img.device, R=RT[:, :3, :3],
                                            T=RT[:, :3, 3], fov=fov)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            v, tri = self.depth2mesh(regressed_pts.view(bs, self.opt.W, self.opt.W), K[:, :3, :3])

        # Regreesed Motion
        # if "train_motion" in self.opt and self.opt.train_motion:
        #       if "motion_model_type" in self.opt and 'resnet' in self.opt.motion_model_type:
        #               flow = self.MotionEncoder(start_img)
        #               flow = self.MotionDecoder(flow)
        #       else:
        #               flow = self.motion_regressor(start_img)
        # else:
        flow_f = batch["motions"][0].cuda()
        flow_p = batch["motions"][1].cuda()
        flow = batch["motions"][0][1:2].cuda().squeeze()

        # forward
        alpha = (1.0 - (middle_index - start_index).float() / (end_index - start_index).float()).view(bs, 1, 1,
                                                                                                      1).cuda()
        Z_f = start_img.new_ones([start_img.shape[0], 1, start_img.shape[2], start_img.shape[3]])  # B, 1, W, W
        Z_f_norm = Z_f - Z_f.max()
        tenInput_f = torch.cat([start_img * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)
        forward_flow = flow_f[middle_index - start_index]  # 0->t

        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            forward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                              self.opt.W)  # 1 3 W W
            forward_pts3D[:, :2, :, :] += forward_flow / (self.opt.W / 2)  # forward_flow: 1, 2 W W
            forward_pts3D = forward_pts3D.view(bs, 3, -1)
            forward_pts3D = forward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_f = self.splatter(forward_pts3D,
                                     tenInput_f.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            forward_v = v.clone()
            forward_v[:, :, :2] += forward_flow.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            textures = Textures(
                    verts_rgb=(tenInput_f).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            meshes = Meshes(verts=forward_v,
                            faces=tri,
                            textures=textures)
            fragments = rasterizer(meshes)
            gen_fs_f = meshes.sample_textures(fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
            #texels = meshes.sample_textures(fragments)
            #blend_params = BlendParams(background_color=[0.0] * tenInput_f.shape[1])
            #gen_fs_f = softmax_rgb_blend(texels, fragments, blend_params, in_channel=tenInput_f.shape[1])[:, :, :, :-1].permute([0, 3, 1, 2])
        else:
            gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                        tenFlow=forward_flow,
                                        tenMetric=start_img.new_ones((start_img.shape[0], 1, start_img.shape[2],
                                                                     start_img.shape[3])))  # B, 65, W, W

        gen_fs = gen_fs_f[:, :-1, :, :]
        tenNormalize = gen_fs_f[:, -1:, :, :]

        # backward
        backward_flow = flow_p[end_index - middle_index]  # 0->-(N-t) 0->t-N
        tenInput_p = torch.cat([end_img * Z_f.exp() * (1 - alpha), Z_f.exp() * (1 - alpha)], 1)
        # 3D Splatter
        if "use_3d_splatter" in self.opt and self.opt.use_3d_splatter:
            backward_pts3D = self.xyzs.view(1, 3, self.opt.W, self.opt.W) * regressed_pts.view(bs, 1, self.opt.W,
                                                                                               self.opt.W)  # 1 3 W W
            backward_pts3D[:, :2, :, :] += backward_flow / (self.opt.W / 2)
            backward_pts3D = backward_pts3D.view(bs, 3, -1)
            backward_pts3D = backward_pts3D.permute(0, 2, 1).contiguous()  # [B, 65536, 3]
            gen_fs_p = self.splatter(backward_pts3D,
                                     tenInput_p.view(bs, -1, self.opt.W * self.opt.W))  # torch.Size([1, 65, 672, 672])
        elif "use_mesh_splatter" in self.opt and self.opt.use_mesh_splatter:
            backward_v = v.clone()
            backward_v[:, :, :2] += backward_flow.permute([0, 2, 3, 1]).view(bs, -1, 2) / (self.opt.W / 2)
            textures = Textures(
                    verts_rgb=(tenInput_p).permute([0, 2, 3, 1]).contiguous().view(bs, -1, tenInput_f.shape[1]))
            meshes = Meshes(verts=backward_v,
                            faces=tri,
                            textures=textures)
            fragments = rasterizer(meshes)
            gen_fs_p = meshes.sample_textures(fragments)[:, :, :, 0, :].permute([0, 3, 1, 2])
            #texels = meshes.sample_textures(fragments)
            #gen_fs_p = softmax_rgb_blend(texels, fragments, blend_params, in_channel=tenInput_p.shape[1])[:, :, :, :-1].permute([0, 3, 1, 2])

        else:
            gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                        tenFlow=backward_flow,
                                        tenMetric=end_img.new_ones(
                                                (end_img.shape[0], 1, end_img.shape[2], end_img.shape[3])))

        gen_fs += gen_fs_p[:, :-1, :, :]
        tenNormalize += gen_fs_p[:, -1:, :, :]

        #tenNormalize[tenNormalize == 0.0] = 1.0
        tenNormalize = torch.clamp(tenNormalize, min=1e-8)
        gen_fs = gen_fs / tenNormalize
        gen_img = gen_fs
        pred_dict = {"PredImg": gen_img,"PredMotion": flow}
        if vis_forward_backward:
            alpha = 1
            tenInput_f = torch.cat([start_img * Z_f_norm.exp() * alpha, Z_f_norm.exp() * alpha], 1)

            forward_flow = flow_f[middle_index - start_index]
            gen_fs_f = self.softsplater(tenInput=tenInput_f,
                                        tenFlow=forward_flow.view(bs, -1, self.opt.W, self.opt.W),
                                        tenMetric=start_img.new_ones((bs, 1, self.opt.W, self.opt.W)))

            gen_fs = gen_fs_f[:, :-1, :, :]
            tenNormalize = gen_fs_f[:, -1:, :, :]
            #tenNormalize[tenNormalize == 0.0] = 1.0
            tenNormalize = torch.clamp(tenNormalize, min=1e-8)
            gen_fs = gen_fs / tenNormalize
            gen_img_forward = gen_fs
            pred_dict["ForawardImg"] = gen_img_forward

            alpha = 0
            tenInput_p = torch.cat([end_img * Z_f.exp() * (1 - alpha), Z_f.exp() * (1 - alpha)], 1)
            backward_flow = flow_p[end_index - middle_index]
            gen_fs_p = self.softsplater(tenInput=tenInput_p,
                                        tenFlow=backward_flow.view(bs, -1, self.opt.W, self.opt.W),
                                        tenMetric=start_img.new_ones((bs, 1, self.opt.W, self.opt.W)))
            gen_fs = gen_fs_p[:, :-1, :, :]
            tenNormalize = gen_fs_p[:, -1:, :, :]
            #tenNormalize[tenNormalize == 0.0] = 1.0
            tenNormalize = torch.clamp(tenNormalize, min=1e-8)
            gen_fs = gen_fs / tenNormalize
            gen_img_backward = gen_fs

            pred_dict["BackwardImg"] = gen_img_backward

        return pred_dict

