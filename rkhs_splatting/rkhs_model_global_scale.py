import torch
import torch.nn  as nn
import numpy as np
import math
from simple_knn._C import distCUDA2
from rkhs_splatting.utils.point_utils import PointCloud
from rkhs_splatting.gauss_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation
from rkhs_splatting.utils.sh_utils import RGB2SH
from icecream import ic
from rkhs_splatting.gauss_model import GaussModel

class RKHSModelGlobalScale(GaussModel):
    """
    The scale of all Gaussians is the same in this model
    """
    
    def __init__(self, sh_degree : int=3, debug=False, trainable=True, scale_trainable=False):
        super(RKHSModelGlobalScale, self).__init__(sh_degree, debug)
        self._trainable = trainable
        self._scale_trainable = scale_trainable

    @property
    def get_scaling(self):
        return self._scaling

    def set_scaling(self, scaling):
        if self._trainable and self._scale_trainable:
            self._scaling = nn.Parameter(scaling)
        else:
            self._scaling = scaling
    
    @property
    def get_features(self):
        return self._features

    def create_from_pcd(self, pcd:PointCloud, initial_scaling=0.005):
        """
            create the guassian model from a color point cloud
        """
        points = pcd.coords
        colors = pcd.select_channels(['R', 'G', 'B'])

        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        # fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())

        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None] # initial scaling
        # scales = torch.sqrt(torch.mean(dist2)) # initial scaling
        scales = torch.scalar_tensor(initial_scaling, device="cuda")
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if self.debug:
            # easy for visualization
            colors = np.zeros_like(colors)
            opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if self._trainable:
            self._xyz = nn.Parameter(fused_point_cloud)
            self._features = nn.Parameter(torch.tensor(np.asarray(colors), device="cuda").float())
            if self._scale_trainable:
                self._scaling = nn.Parameter(scales)
            else:
                self._scaling = scales
            self._opacity = nn.Parameter(opacities)
        else:
            self._xyz = fused_point_cloud
            self._features = torch.tensor(np.asarray(colors), device="cuda").float()
            self._scaling = scales
            self._opacity = opacities
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        return self

    def to_pc(self):
        N = self._xyz.shape[0]
        pc_coords = self._xyz.detach().cpu().numpy()
        pc_rgbs = self._features.detach().cpu().numpy()
        pc_channels = dict(
            R = pc_rgbs[:,0],
            G = pc_rgbs[:,1],
            B = pc_rgbs[:,2],
            A = self.get_opacity.detach().cpu().numpy().flatten()
        )
        pc = PointCloud(pc_coords, pc_channels)
        return pc