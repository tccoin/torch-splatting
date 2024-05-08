import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from icecream import ic

# from torch.profiler import profile, ProfilerActivity

# USE_PROFILE = False


# if USE_PROFILE:
#     prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
# else:
#     prof = contextlib.nullcontext()

class SL1Loss(nn.Module):
    def __init__(self, ohem=False, topk=0.6):
        super(SL1Loss, self).__init__()
        self.ohem = ohem
        self.topk = topk
        self.loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, inputs, targets, mask):
        loss = self.loss(inputs[mask], targets[mask])

        if self.ohem:
            num_hard_samples = int(self.topk * loss.numel())
            loss, _ = torch.topk(loss.flatten(), 
                                 num_hard_samples)

        return torch.mean(loss)


def rkhs_global_scale_loss(prediction_tiles, gt_tiles, gt_rgb, scale3d, use_geometry=True, use_rgb=True):

    mean2d_tiles = prediction_tiles['mean2d']
    mean3d_tiles = prediction_tiles['mean3d']
    scale2d_tiles = prediction_tiles['scale2d']
    label_tiles = prediction_tiles['label']

    gt_mean2d_tiles = gt_tiles['mean2d']
    gt_mean3d_tiles = gt_tiles['mean3d']
    gt_label_tiles = gt_tiles['label']

    N = len(mean2d_tiles)
    M = len(mean2d_tiles[0])
    T = gt_rgb.shape[0]//N

    # ic(scale3d)

    scale3d_squared = scale3d**2

    import numpy as np
    mean_tile_number = np.mean([scale2d_tiles[v][u].shape[0] for v in range(N) for u in range(M)])



    # local map norm, training image norm, inner product
    loss = [0, 0, 0]
    init = False
    for v in range(N):
        for u in range(M):
            
            B = mean2d_tiles[v][u].shape[0]
            if B == 0 and init:
                continue
            init = True

            # pc_tile = gt_points[v][u]
            # gt_rgb_tile = torch.from_numpy(pc_tile.select_channels(['R', 'G', 'B'])/255.0).to(scale3d.device)
            # gt_points_tile = torch.from_numpy(pc_tile.coords).to(scale3d.device)
            gt_rgb_tile = gt_label_tiles[v][u][0]
            gt_mean3d_tile = gt_mean3d_tiles[v][u]
            P = gt_rgb_tile.shape[0]

            # ic(M, N, B, P, T)

            gt_rgb_tile_unsq0 = gt_rgb_tile.unsqueeze(0)
            gt_rgb_tile_unsq1 = gt_rgb_tile.unsqueeze(1)
            gt_mean3d_tile_unsq0 = gt_mean3d_tile.unsqueeze(0)
            gt_mean3d_tile_unsq1 = gt_mean3d_tile.unsqueeze(1)

            rgb_tile = label_tiles[v][u][0]
            opacity_tile = label_tiles[v][u][2]
            rgb_tile_unsq0 = rgb_tile.unsqueeze(0)
            rgb_tile_unsq1 = rgb_tile.unsqueeze(1)
            mean3d_tile = mean3d_tiles[v][u]
            mean3d_tile_unsq0 = mean3d_tile.unsqueeze(0)
            mean3d_tile_unsq1 = mean3d_tile.unsqueeze(1)

            # ic(mean3d_tile.shape)
            # ic(rgb_tile.shape)
            # if rgb_tile.shape[0]>0:
            #     ic(torch.min(rgb_tile), torch.max(rgb_tile))

            scale_rgb_squared = 2**2 # 2
            # scale3d_squared = 3**2 #0.015**2
            geo_cut_off = exp(-0.5 *(0.1**2)*3/scale3d_squared)

            # 0: local map inner product
            # 1: current frame inner product
            # 2: inner product between local map and current frame

            # ic(M, N, B, P, T)

            if use_rgb:
                rgb0 = (-0.5 * (rgb_tile_unsq1 - rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                rgb1 = (-0.5 * (gt_rgb_tile_unsq1 - gt_rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                rgb2 = (-0.5 * (rgb_tile_unsq1 - gt_rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                # rgb0 = torch.where(rgb0 < rgb_cut_off, 1, rgb0)
                # rgb1 = torch.where(rgb1 < rgb_cut_off, 1, rgb1)
                # rgb2 = torch.where(rgb2 < rgb_cut_off, 1, rgb2)
            else:
                rgb0 = 1
                rgb1 = 1
                rgb2 = 1
            if use_geometry:
                geo0 = (-0.5 * (mean3d_tile_unsq1 - mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_squared).exp()
                geo1 = (-0.5 * (gt_mean3d_tile_unsq1 - gt_mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_squared).exp()
                geo2 = (-0.5 * (mean3d_tile_unsq1 - gt_mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_squared).exp()

                # ic(mean3d_tile_unsq1.shape, mean3d_tile_unsq0.shape)
                # geo0 = torch.where(geo0 < geo_cut_off, 1, geo0)
                # geo1 = torch.where(geo1 < geo_cut_off, 1, geo1)
                # geo2 = torch.where(geo2 < geo_cut_off, 1, geo2)
            else:
                geo0 = 1
                geo1 = 1
                geo2 = 1

            loss0 = rgb0 * geo0
            loss1 = rgb1 * geo1
            loss2 = rgb2 * geo2

            # naive way to reduce opacity in empty space
            # k = torch.tensor(geo2.sum(axis=1)).requires_grad_(False)
            # loss_opacity = torch.where(k<0.001, opacity_tile.squeeze(1)*1e6/(k+1e-6), 0)
            # loss[0] += loss_opacity.sum()
            # ic(opacity_tile.max(), opacity_tile.min())
            # ic(k.shape, opacity_tile.shape, loss_opacity.shape)

            loss[0] = loss0.sum() + loss[0]
            loss[1] = loss1.sum() + loss[1]
            loss[2] = loss2.sum() + loss[2]
    return loss

def l1_loss(prediction, gt):
    return torch.abs((prediction - gt)).mean()

def l2_loss(prediction, gt):
    return ((prediction - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)