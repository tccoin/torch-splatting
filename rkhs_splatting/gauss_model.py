import torch
import torch.nn  as nn
import numpy as np
import math
from simple_knn._C import distCUDA2
from gaussian_splatting.utils.point_utils import PointCloud
from gaussian_splatting.gauss_render import strip_symmetric, inverse_sigmoid, build_scaling_rotation
from gaussian_splatting.utils.sh_utils import RGB2SH
from rkhs_splatting.utils.trainer_utils import get_expon_lr_func
from icecream import ic

class GaussModel(nn.Module):
    """
    A Gaussian Model

    * Attributes
    _id: id of the model
    _xyz: locations of gaussians
    _feature_dc: DC term of features
    _feature_rest: rest features
    _rotatoin: rotation of gaussians
    _scaling: scaling of gaussians
    _opacity: opacity of gaussians

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.opt_parameters = {}
        self.opt_lr = {}
    
    def __init__(self, sh_degree=3, debug=False, **kwargs):
        super(GaussModel, self).__init__()
        self.max_sh_degree = sh_degree  
        self._id = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.debug = debug

    def update_learning_rate(self, new_lr):
        if new_lr is not None:
            for group_name, lr_value in new_lr.items():
                if type(lr_value) == dict:
                    lr_value['scheduler'] = get_expon_lr_func(
                        lr_init=lr_value['lr_init'],
                        lr_final=lr_value['lr_final'],
                        lr_delay_steps=lr_value['lr_delay_steps'],
                        lr_delay_mult=lr_value['lr_delay_mult'],
                        max_steps=lr_value['max_steps']
                    )
                self.opt_lr[group_name] = lr_value

    def set_learning_rate(self, optimizer, step):
        for param_group in optimizer.param_groups:
            group_name = param_group['name']
            if group_name in self.opt_lr:
                lr_value = self.opt_lr[group_name]
                if type(lr_value)==dict:
                    lr_value = lr_value['scheduler'](step)
                param_group['lr'] = lr_value
                # print("Setting learning rate for group {} to {}".format(group_name, lr_value))
            else:
                print("Warning: No learning rate found for group {}".format(group_name))

    def create_from_pcd(self, pcd:PointCloud):
        """
            create the guassian model from a color point cloud
        """
        points = pcd.coords
        colors = pcd.select_channels(['R', 'G', 'B'])

        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        if self.debug:
            # easy for visualization
            colors = np.zeros_like(colors)
            opacities = inverse_sigmoid(0.9 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._id = torch.arange(fused_point_cloud.shape[0], device="cuda")
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.count = torch.zeros((self._xyz.shape[0]), device="cuda")
        return self

    @property
    def get_ids(self):
        return self._id

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_count(self):
        return self.count

    def add_count(self, mask):
        self.count[mask] += 1
    
    def reset_id_and_count(self):
        self._id = torch.arange(self._xyz.shape[0], device="cuda")
        self.count = torch.zeros_like(self._id, device="cuda")
    
    def prune_points(self, mask):
        # self._id = self._id[mask]
        # self.count = self.count[mask]
        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        self._opacity = self._opacity[mask]
        self.max_radii2D = self.max_radii2D[mask]

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement
        # import os
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
