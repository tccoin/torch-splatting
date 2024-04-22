import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from icecream import ic

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

pix_coord = torch.stack(torch.meshgrid(torch.arange(512), torch.arange(512), indexing='xy'), dim=-1).to('cuda')

def rkhs_global_scale_loss(prediction_tiles, gt_points, gt_rgb, scale3d):
    loss = torch.zeros(3, device=scale3d.device) # local map norm, training image norm, inner product

    mean2d_tile = prediction_tiles['mean2d']
    mean3d_tile = prediction_tiles['mean3d']
    scale2d_tile = prediction_tiles['scale2d']
    label_tile = prediction_tiles['label']

    N = len(mean2d_tile)
    M = len(mean2d_tile[0])
    if type(mean2d_tile[0][0]) is dict:
        return loss
    T = mean2d_tile[0][0].shape[0]

    gt_points = gt_points.random_sample(10)
    gt_label_tile = torch.from_numpy(gt_points.select_channels(['R', 'G', 'B'])).to(scale3d.device)
    gt_points_tile = torch.from_numpy(gt_points.coords).to(scale3d.device)
    scale3d_squared = scale3d**2

    import numpy as np
    mean_tile_number = np.mean([scale2d_tile[v][u].shape[0] for v in range(N) for u in range(M)])
    ic(mean_tile_number)

    for v in range(N):
        for u in range(M):
            # 2d
            # tile_coord = pix_coord[T*v:T*(v+1), T*u:T*(u+1)].reshape(-1, 2) # T**2, 2
            # scale2d_squared = torch.max(scale2d_tile[v][u], dim=-1)**2 # B
            # loss0 = mean2d_tile[v][u].unsqueeze(1) - tile_coord.unsqueeze(0) # B, T**2, 2
            # loss0 = -0.5*torch.norm(loss0, p=2, dim=2)**2 # B, T**2
            # loss[0] += torch.exp(loss0/scale2d[0]).mean()

            # 3d
            # gt_points: T**2, 3

            # get_tile_data = lambda x: torch.from_numpy(x[T*v:T*(v+1), T*u:T*(u+1)]).to(scale3d.device)
            # gt_label_tile = get_tile_data(gt_points.select_channels(['R', 'G', 'B'])) # T**2, 3, only color now
            # gt_points_tile = get_tile_data(gt_points.coords) # T**2, 3

            rgb_loss0 = label_tile[v][u][0].unsqueeze(1) - label_tile[v][u][0].unsqueeze(0) # B, B, 3
            rgb_loss0 = torch.norm(rgb_loss0, p=2, dim=2)**2 # B, B
            loss0 = mean3d_tile[v][u].unsqueeze(1) - mean3d_tile[v][u].unsqueeze(0) # B, B, 3
            loss0 = torch.exp(-0.5 * torch.norm(loss0, p=3, dim=2)**2 / scale3d_squared) # B, B
            loss0 = rgb_loss0*loss0

            rgb_loss1 = gt_label_tile.unsqueeze(1) - gt_label_tile.unsqueeze(0) # B, T**2, 3
            rgb_loss1 = torch.norm(rgb_loss1, p=2, dim=2)**2 # B, T**2
            loss1 = gt_points_tile.unsqueeze(1) - gt_points_tile.unsqueeze(0) # B, T**2, 3
            loss1 = torch.exp(-0.5 * torch.norm(loss1, p=3, dim=2)**2 / scale3d_squared) # B, T**2
            loss1 = rgb_loss1*loss1

            rgb_loss2 = label_tile[v][u][0].unsqueeze(1) - gt_label_tile.unsqueeze(0) # B, T**2, 3
            rgb_loss2 = torch.norm(rgb_loss2, p=2, dim=2)**2 # B, T**2
            loss2 = mean3d_tile[v][u].unsqueeze(1) - gt_points_tile.unsqueeze(0) # B, T**2, 3
            loss2 = torch.exp(-0.5 * torch.norm(loss2, p=3, dim=2)**2 / scale3d_squared) # B, T**2
            loss2 = rgb_loss2*loss2

            loss[0] += loss0.mean()
            loss[1] += loss1.mean()
            loss[2] += loss2.mean()

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