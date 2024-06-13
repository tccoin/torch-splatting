import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from icecream import ic
import numpy as np
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


def rkhs_loss_global_scale(prediction_tiles, gt_tiles, gt_rgb, scale3d, use_geometry=True, use_rgb=True):

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

    scale_rgb_squared = 0.03
    scale3d_squared = scale3d**2
    # geo_cut_off = exp(-0.5 *9)

    # local map norm, training image norm, inner product
    loss = [torch.tensor([0.]).requires_grad_(True).cuda() for i in range(3)]
    empty2d = torch.empty(0,2,device='cuda')
    inner_product_tiles = {v:{u:empty2d for u in range(M)} for v in range(N)} # h,w,b,p
    
    n_loss0 = 0
    n_loss1 = 0
    n_loss2 = 0
    
    for v in range(N):
        for u in range(M):

            gt_rgb_tile = gt_label_tiles[v][u][0]
            gt_mean3d_tile = gt_mean3d_tiles[v][u]
            P = gt_rgb_tile.shape[0] # train point size
            B = mean2d_tiles[v][u].shape[0] # map point size
            if B == 0 or P==0:
                continue
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

            # 0: local map inner product
            # 1: current frame inner product
            # 2: inner product between local map and current frame

            if use_rgb:
                rgb0 = (-0.5 * (rgb_tile_unsq1 - rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                rgb1 = (-0.5 * (gt_rgb_tile_unsq1 - gt_rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                rgb2 = (-0.5 * (rgb_tile_unsq1 - gt_rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                # rgb0 = torch.where(rgb0 < rgb_cut_off, 0, rgb0)
                # rgb1 = torch.where(rgb1 < rgb_cut_off, 0, rgb1)
                # rgb2 = torch.where(rgb2 < rgb_cut_off, 0, rgb2)
            else:
                rgb0 = 1
                rgb1 = 1
                rgb2 = 1
            if use_geometry:
                geo0 = (-0.5 * (mean3d_tile_unsq1 - mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_squared).exp()
                geo1 = (-0.5 * (gt_mean3d_tile_unsq1 - gt_mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_squared).exp()
                geo2 = (-0.5 * (mean3d_tile_unsq1 - gt_mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_squared).exp()

                # ic(mean3d_tile_unsq1.shape, mean3d_tile_unsq0.shape)
                # geo0 = torch.where(geo0 < geo_cut_off, 0, geo0)
                # geo1 = torch.where(geo1 < geo_cut_off, 0, geo1)
                # geo2 = torch.where(geo2 < geo_cut_off, 0, geo2)
            else:
                geo0 = 1
                geo1 = 1
                geo2 = 1

            loss0 = rgb0 * geo0
            loss1 = rgb1 * geo1
            loss2 = rgb2 * geo2

            inner_product_tiles[v][u] = geo2

            n_loss0 += loss0.numel()
            n_loss1 += loss1.numel()
            n_loss2 += loss2.numel()

            loss[0] = loss0.sum() + loss[0]
            loss[1] = loss1.sum() + loss[1]
            loss[2] = loss2.sum() + loss[2]

    # ic(loss, n_loss0, n_loss1, n_loss2)

    loss[0] /= n_loss0
    loss[1] /= n_loss1
    loss[2] /= n_loss2

    # ic(loss)


    return loss, inner_product_tiles


def rkhs_loss(prediction_tiles, gt_tiles, gt_rgb, scale3d, use_geometry=True, use_rgb=True):

    mean2d_tiles = prediction_tiles['mean2d']
    mean3d_tiles = prediction_tiles['mean3d']
    scale2d_tiles = prediction_tiles['scale2d']
    scale3d_tiles = prediction_tiles['scale3d']
    label_tiles = prediction_tiles['label']

    gt_mean2d_tiles = gt_tiles['mean2d']
    gt_mean3d_tiles = gt_tiles['mean3d']
    gt_label_tiles = gt_tiles['label']

    N = len(mean2d_tiles)
    M = len(mean2d_tiles[0])
    T = gt_rgb.shape[0]//N

    # ic(scale3d)

    scale_rgb_squared = 4
    # scale3d_squared = 0.01**2
    # geo_cut_off = exp(-0.5 *9)

    # local map norm, training image norm, inner product
    loss = [torch.tensor([0.]).requires_grad_(True).cuda() for i in range(3)]
    empty2d = torch.empty(0,2,device='cuda')
    inner_product_tiles = {v:{u:empty2d for u in range(M)} for v in range(N)} # h,w,b,p

    n_loss0 = 0
    n_loss1 = 0
    n_loss2 = 0

    for v in range(N):
        for u in range(M):

            gt_rgb_tile = gt_label_tiles[v][u][0]
            gt_mean3d_tile = gt_mean3d_tiles[v][u]
            P = gt_rgb_tile.shape[0] # train point size
            B = mean2d_tiles[v][u].shape[0] # map point size
            if B == 0 or P==0:
                continue
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
            scale3d_tile = scale3d_tiles[v][u]

            # 0: local map inner product
            # 1: current frame inner product
            # 2: inner product between local map and current frame

            if use_rgb:
                rgb0 = (-0.5 * (rgb_tile_unsq1 - rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                rgb1 = (-0.5 * (gt_rgb_tile_unsq1 - gt_rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                rgb2 = (-0.5 * (rgb_tile_unsq1 - gt_rgb_tile_unsq0).pow(2).sum(-1) / scale_rgb_squared).exp()
                # rgb0 = torch.where(rgb0 < rgb_cut_off, 0, rgb0)
                # rgb1 = torch.where(rgb1 < rgb_cut_off, 0, rgb1)
                # rgb2 = torch.where(rgb2 < rgb_cut_off, 0, rgb2)
            else:
                rgb0 = 1
                rgb1 = 1
                rgb2 = 1
            if use_geometry:

                # ic(scale3d_tile.shape)
                scale3d_tile_squared = scale3d_tile.reshape(-1,1)**2

                # tmp = (mean3d_tile_unsq1 - mean3d_tile_unsq0).pow(2).sum(-1)
                # ic(tmp.shape, tmp)
                # ic(scale3d_tile_squared.shape)
                # scale3d_tile_squared = scale3d_tile_squared
                # ic(scale3d_tile_squared)
                # ic(tmp / scale3d_tile_squared)

                geo0 = (-0.5 * (mean3d_tile_unsq1 - mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_tile_squared).exp()
                geo2 = (-0.5 * (mean3d_tile_unsq1 - gt_mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_tile_squared).exp()
                scale3d_tile_squared = scale3d_tile.mean()**2
                geo1 = (-0.5 * (gt_mean3d_tile_unsq1 - gt_mean3d_tile_unsq0).pow(2).sum(-1) / scale3d_tile_squared).exp()

                # ic(mean3d_tile_unsq1.shape, mean3d_tile_unsq0.shape)
                # geo0 = torch.where(geo0 < geo_cut_off, 0, geo0)
                # geo1 = torch.where(geo1 < geo_cut_off, 0, geo1)
                # geo2 = torch.where(geo2 < geo_cut_off, 0, geo2)
            else:
                geo0 = 1
                geo1 = 1
                geo2 = 1

            loss0 = rgb0 * geo0
            loss1 = rgb1 * geo1
            loss2 = rgb2 * geo2

            inner_product_tiles[v][u] = geo2

            n_loss0 += loss0.numel()
            n_loss1 += loss1.numel()
            n_loss2 += loss2.numel()

            loss[0] = loss0.sum() + loss[0]
            loss[1] = loss1.sum() + loss[1]
            loss[2] = loss2.sum() + loss[2]

    loss[0] /= n_loss0
    loss[1] /= n_loss1
    loss[2] /= n_loss2

    return loss, inner_product_tiles


def check_rkhs_loss(n_points, id_tile, inner_product_tiles):
    N = len(inner_product_tiles)
    M = len(inner_product_tiles[0])
    map_point_scores = torch.zeros(n_points, device='cuda')
    for v in range(N):
        for u in range(M):
            inner_product = inner_product_tiles[v][u]
            ids = id_tile[v][u]
            B, P = inner_product.shape # map/train point size
            tile_point_scores = inner_product.sum(dim=1)
            for i in range(B):
                map_point_scores[ids[i]] += tile_point_scores[i]
    return map_point_scores

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