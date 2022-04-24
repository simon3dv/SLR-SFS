import torch
import torch.nn as nn
import random
import copy
import numpy as np

def euler_integration(motion, destination_frame, return_all_frames=False):
    """
    This function is provided by Aleksander Hołyński <holynski@cs.washington.edu>
    Repeatedly integrates the Eulerian motion field to get the displacement map to a future frame.

    :param motion: The Eulerian motion field to be integrated.
    :param destination_frame: The number of times the motion field should be integrated.
    :param return_all_frames: Return the displacement maps to all intermediate frames, not only the last frame.
    :return: The displacement map resulting from repeated integration of the motion field.
    """

    assert (motion.dim() == 4)
    b, c, height, width = motion.shape
    assert (b == 1), 'Function only implemented for batch = 1'
    assert (c == 2), f'Input motion field should be Bx2xHxW. Given tensor is: {motion.shape}'

    y, x = torch.meshgrid(
        [torch.linspace(0, height - 1, height, device='cuda'),
         torch.linspace(0, width - 1, width, device='cuda')])
    coord = torch.stack([x, y], dim=0).long()

    destination_coords = coord.clone().float()
    if return_all_frames:
        displacements = torch.zeros(destination_frame + 1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(b + 1, 1, height, width, device='cuda')
    else:
        displacements = torch.zeros(1, 2, height, width, device='cuda')
        visible_pixels = torch.ones(1, 1, height, width, device='cuda')
    invalid_mask = torch.zeros(1, height, width, device='cuda').bool()
    for frame_id in range(1, destination_frame + 1):
        destination_coords = destination_coords + motion[0][:, torch.round(destination_coords[1]).long(),
                                                  torch.round(destination_coords[0]).long()]
        out_of_bounds_x = torch.logical_or(destination_coords[0] > (width - 1), destination_coords[0] < 0)
        out_of_bounds_y = torch.logical_or(destination_coords[1] > (height - 1), destination_coords[1] < 0)
        invalid_mask = torch.logical_or(out_of_bounds_x.unsqueeze(0), invalid_mask)
        invalid_mask = torch.logical_or(out_of_bounds_y.unsqueeze(0), invalid_mask)

        # Set the displacement of invalid pixels to zero, to avoid out-of-bounds access errors
        destination_coords[invalid_mask.expand_as(destination_coords)] = coord[
            invalid_mask.expand_as(destination_coords)].float()
        if return_all_frames:
            displacements[frame_id] = (destination_coords - coord.float()).unsqueeze(0)
            # Set the displacements for invalid pixels to be out of bounds.
            displacements[frame_id][invalid_mask] = torch.max(height, width) + 1
            visible_pixels[frame_id] = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
        else:
            displacements = (destination_coords - coord.float()).unsqueeze(0)
            visible_pixels = torch.logical_not(invalid_mask.clone()).float().unsqueeze(0)
            displacements[invalid_mask.unsqueeze(0).repeat(1,2,1,1)] = torch.max(torch.Tensor([height, width])) + 1
    return displacements,visible_pixels

class EulerIntegration(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
    def forward(self, motion, destination_frame, return_all_frames=False,show_visible_pixels=False):
        displacements = torch.zeros(motion.shape).to(motion.device)
        visible_pixels = torch.zeros(motion.shape[0], 1, motion.shape[2], motion.shape[3])
        for b in range(motion.shape[0]):
            displacements[b:b+1], visible_pixels[b:b+1] = euler_integration(motion[b:b+1], destination_frame[b])

        if show_visible_pixels:
            return displacements, visible_pixels
        else:
            return displacements



'''
class EulerIntegration(nn.Module):
    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.xs = torch.linspace(0, self.opt.W - 1, self.opt.W).cuda()
        self.ys = torch.linspace(0, self.opt.W - 1, self.opt.W).cuda()
        self.xs = self.xs.view(1, 1, self.opt.W).repeat(1, self.opt.W, 1)
        self.ys = self.ys.view(1, self.opt.W, 1).repeat(1, 1, self.opt.W)
        self.zeros = torch.zeros((2, self.opt.W, self.opt.W)).view(1, 2, -1).repeat(1, 1, 1)
    def forward(
            self, flow, time):
        """
        flow:torch.Size([bs, 2 or 3, W, W])
        """
        bs = flow.shape[0]
        flow_f = torch.zeros((bs, 2, self.opt.W, self.opt.W)).view(bs, 2, -1).cuda()
        M = flow[:,:2,...].clone().view(bs, 2, -1)
        flow_f = flow.clone().view(bs, 2, -1)
        out_points = torch.zeros(bs, self.opt.W * self.opt.W).cuda()
        new_xy_l = torch.zeros(bs, self.opt.W * self.opt.W).long().cuda()
        x = self.xs.clone().view(1, -1).repeat(bs, 1)  # bs, W*W
        y = self.ys.clone().view(1, -1).repeat(bs, 1)  # bs, W*W
        for b in range(0, bs):
            if time[b] == 0:
                flow_f[b] = torch.zeros((2, self.opt.W*self.opt.W))
            elif time[b] == 1:
                flow_f[b] = flow[b, :2,...].clone().view(2, -1)
            else:
                M[b] = flow[b,:2,...].clone().view(2, -1)
                flow_f[b] = flow[b, :2,...].clone().view(2, -1)
                for t in range(2, time[b]+1):
                    x[b] = x[b].to(M[b].device) + M[b, 0, :]
                    y[b] = y[b].to(M[b].device) + M[b, 1, :]
                    out_points[b] = (((x[b] < 0).long() + (x[b] >= self.opt.W).long() + \
                                   (y[b] < 0).long() + (y[b] >= self.opt.W).long()) > 0).float().view(-1)  # bs, W*W
                    new_xy_l[b] = torch.clamp(x[b].long() + y[b].long() * self.opt.W, 0, self.opt.W * self.opt.W - 1).long()  # W*W
                    M[b] = flow[b, :2, new_xy_l[b].clone()] * (1 - out_points[b])  # 2,W*W
                    flow_f[b] = flow_f[b] + M[b]
        flow_f = flow_f.view(bs, 2, self.opt.W, self.opt.W)
        return flow_f
    def forward_all(self, flow, time):
        """
        flow:torch.Size([2, W, W])
        """
        flow_f = torch.zeros((time, 2, self.opt.W, self.opt.W)).view(time, 2, -1).cuda()
        flow_f[0] = torch.zeros((2, self.opt.W * self.opt.W))
        flow_f[1] = flow.clone().view(2, -1)
        x = self.xs.clone().view(-1)  # W*W
        y = self.ys.clone().view(-1)  # W*W
        M = flow.clone().view(2, -1)
        for t in range(2, time):
            x = x + M[0, :]  # W*W
            y = y + M[1, :]
            out_points = (((x < 0).long() + (x >= self.opt.W).long() + \
                    (y < 0).long() + (y >= self.opt.W).long()) > 0).float().view(-1)  # 1, W*W
            new_xy_l = torch.clamp(x.long() + y.long() * self.opt.W, 0, self.opt.W * self.opt.W - 1).long()  # W*W
            M = flow[:, new_xy_l] * (1 - out_points)  # 2,W*W
            flow_f[t] = flow_f[t-1] + M
        flow_f = flow_f.view(time, 2, self.opt.W    , self.opt.W)
        return flow_f
'''