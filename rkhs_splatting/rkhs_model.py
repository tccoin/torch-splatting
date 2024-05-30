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

class RKHSModel(GaussModel):
    """
    The scale of all Gaussians is the same in this model
    """

    def __init__(self, sh_degree : int=3, debug=False, trainable=True, scale_trainable=False):
        super(RKHSModel, self).__init__(sh_degree, debug)
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

    def prune_points(self, mask, optimizer=None):
        mask = mask.cuda()
        if self._trainable:
            # update optimizer
            new_parameters = {}
            N = self._xyz.shape[0]
            for group in optimizer.param_groups:
                if group['params'][0].shape[0] != N:
                    continue
                # apply mask to the parameter
                group['params'][0] = nn.Parameter(group['params'][0][mask])
                new_parameters[group['name']] = group['params'][0]
                # apply mask to the optimizer state
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                    optimizer.state[group['params'][0]] = stored_state
            # update model
            self._xyz = new_parameters['xyz']
            self._features = new_parameters['features']
            self._opacity = new_parameters['opacity']
        else:
            self._xyz = self._xyz[mask]
            self._features = self._features[mask]
            self._opacity = self._opacity[mask]

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
        scales = initial_scaling * torch.ones((fused_point_cloud.shape[0]), device="cuda")
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if self.debug:
            # easy for visualization
            colors = np.zeros_like(colors)
            opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._id = torch.arange(fused_point_cloud.shape[0], device="cuda")
        parameters = []
        if self._trainable:
            self._xyz = nn.Parameter(fused_point_cloud)
            self._features = nn.Parameter(torch.tensor(np.asarray(colors), device="cuda").float())
            self._opacity = nn.Parameter(opacities)
            parameters.append({'name': 'xyz', 'params': [self._xyz]})
            parameters.append({'name': 'features', 'params': [self._features]})
            parameters.append({'name': 'opacity', 'params': [self._opacity]})
            if self._scale_trainable:
                self._scaling = nn.Parameter(scales)
                parameters.append({'name': 'scaling', 'params': [self._scaling]})
            else:
                self._scaling = scales
        else:
            self._xyz = fused_point_cloud
            self._features = torch.tensor(np.asarray(colors), device="cuda").float()
            self._opacity = opacities
            self._scaling = scales
        self.count = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.opt_parameters = parameters

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
