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
    
    def __init__(self, sh_degree : int=3, debug=False, trainable=True, scale_trainable=True):
        super(RKHSModel, self).__init__(sh_degree, debug)
        self._trainable = trainable
        self._scale_trainable = scale_trainable
        self.count = torch.empty(0)
        self.grad_sum = torch.empty(0)
        self.grad_update_count = torch.empty(0)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    def set_scaling(self, scaling):
        if self._trainable and self._scale_trainable:
            self._scaling = nn.Parameter(torch.log(scaling))
        else:
            self._scaling = torch.log(scaling)
    
    @property
    def get_features(self):
        return torch.sigmoid(self._features)
    
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def create_from_pcd(
            self,
            pcd:PointCloud,
            initial_scaling=0.005,
            xyz_lr_init=1e-3,
            xyz_lr_final=1e-4,
            xyz_lr_delay_multi=0.01,
            xyz_lr_max_steps=10000,
            features_lr=3e-3,
            opacity_lr=5e-2,
            scaling_lr=5e-3
        ):
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
        scales = torch.log(torch.sqrt(dist2)) # initial scaling
        # scales = torch.log(initial_scaling * torch.ones((fused_point_cloud.shape[0]), device="cuda"))
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1
        opacities = inverse_sigmoid(0.95 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        parameters = []
        if self._trainable:
            self._xyz = nn.Parameter(fused_point_cloud)
            self._features = nn.Parameter(inverse_sigmoid(torch.tensor(np.asarray(colors), device="cuda").float()))
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
            self._features = inverse_sigmoid(torch.tensor(np.asarray(colors), device="cuda").float())
            self._opacity = opacities
            self._scaling = scales

        self.opt_parameters = parameters
        self.update_learning_rate({
            'xyz': {
                'lr_init': 1e-2,
                'lr_final': 1e-4,
                'lr_delay_steps': 0,
                'lr_delay_mult': 1,
                'max_steps': 10000
            },
            'features': 3e-3,
            'opacity': 3e-3,
            'scaling': 1e-5
        })

        self._id = torch.arange(fused_point_cloud.shape[0], device="cuda")
        self.count = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.reset_densification_stats()

        return self

    def save_to(self, path, note=''):
        if note != '':
            note = ', ' + note
        header = f"x, y, z, r, g, b, opacity, scaling\n"
        with open(path, 'w+') as f:
            f.write(header)
            for i in range(self.get_xyz.shape[0]):
                for j in range(3):
                    f.write(f"{self.get_xyz[i,j].item()},")
                for j in range(3):
                    f.write(f"{self.get_features[i,j].item()},")
                f.write(f"{self.get_opacity[i].item()},{self.get_scaling[i].item()}\n")


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

    def replace_tensor_to_optimizer(self, tensor, name, optimizer):
        optimizable_tensors = {}
        for group in optimizer.param_groups:
            if group["name"] == name:
                stored_state = optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor)
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def densification_postfix(
            self,
            new_points_dict,
            optimizer
        ):
        # apply inverse activation function
        new_points_dict['scaling'] = torch.log(new_points_dict['scaling'])
        # update parameters
        if self._trainable:
            # update optimizer
            new_parameters = {}
            N = self._xyz.shape[0]
            for group in optimizer.param_groups:
                if group['params'][0].shape[0] != N:
                    continue
                # apply mask to the optimizer parameters
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state['exp_avg'] = torch.cat([stored_state['exp_avg'], torch.zeros_like(new_points_dict[group['name']])])
                    stored_state['exp_avg_sq'] = torch.cat([stored_state['exp_avg_sq'], torch.zeros_like(new_points_dict[group['name']])])
                    del optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter(torch.cat([group['params'][0], new_points_dict[group['name']]]))
                    optimizer.state[group['params'][0]] = stored_state
                    new_parameters[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(torch.cat([group['params'][0], new_points_dict[group['name']]]))
                    new_parameters[group['name']] = group['params'][0]

            # update model
            self._xyz = new_parameters['xyz']
            self._features = new_parameters['features']
            self._opacity = new_parameters['opacity']
            if self._scale_trainable:
                self._scaling = new_parameters['scaling']
            else:
                self._scaling = torch.cat([self._scaling, new_points_dict['scaling']])
        else:
            self._xyz = torch.cat([self._xyz, new_points_dict['xyz']])
            self._scaling = torch.cat([self._scaling, new_points_dict['scaling']])
            self._features = torch.cat([self._features, new_points_dict['features']])
            self._opacity = torch.cat([self._opacity, new_points_dict['opacity']])
        
        self.count = torch.cat([self.count, torch.zeros_like(new_points_dict['xyz'][:,0])])
        self.grad_sum = torch.cat([self.grad_sum, torch.zeros_like(new_points_dict['xyz'][:,0])])
        self.grad_update_count = torch.cat([self.grad_update_count, torch.zeros_like(new_points_dict['xyz'][:,0])])
        self._id = torch.arange(self.count.shape[0], device="cuda")
    
    def prune_points(self, mask, optimizer):
        """
        mask: 1 to keep, 0 to prune
        """
        mask = mask.cuda()
        if self._trainable:
            # update optimizer
            new_parameters = {}
            N = self._xyz.shape[0]
            for group in optimizer.param_groups:
                if group['params'][0].shape[0] != N:
                    continue
                # apply mask to the optimizer parameters
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                    stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                    del optimizer.state[group['params'][0]]
                    group['params'][0] = nn.Parameter(group['params'][0][mask])
                    optimizer.state[group['params'][0]] = stored_state
                    new_parameters[group['name']] = group['params'][0]
                else:
                    group['params'][0] = nn.Parameter(group['params'][0][mask])
                    new_parameters[group['name']] = group['params'][0]
            # update model
            self._xyz = new_parameters['xyz']
            self._features = new_parameters['features']
            self._opacity = new_parameters['opacity']
            if self._scale_trainable:
                self._scaling = new_parameters['scaling']
            else:
                self._scaling = self._scaling[mask]
        else:
            self._xyz = self._xyz[mask]
            self._features = self._features[mask]
            self._opacity = self._opacity[mask]
            self._scaling = self._scaling[mask]
        self.count = self.count[mask]
        self.grad_sum = self.grad_sum[mask]
        self.grad_update_count = self.grad_update_count[mask]
        self._id = torch.arange(self.count.shape[0], device="cuda")

    def add_densification_stats(self):
        curr_grad = self.get_xyz.grad.norm(dim=-1)
        self.grad_sum += curr_grad
        self.grad_update_count += 1

    def reset_densification_stats(self):
        self.grad_sum = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.grad_update_count = torch.zeros((self._xyz.shape[0]), device="cuda")

    
    def densify(
            self,
            optimizer,
            world_extent=1, #tartanair 5
            max_screen_size=20,
            grad_threshold=1e-3,#tartanair 2e-4
            dense_percent=0.01,
            n_repeat=2,
        ):
        # mask
        avg_grad = self.grad_sum/self.grad_update_count #optimizer.param_groups[0]['lr']
        ic(avg_grad.median(), avg_grad.mean(), avg_grad.max(), avg_grad.min())
        # dense_threshold = 0
        dense_threshold = dense_percent * world_extent
        large_grad_mask = avg_grad > grad_threshold
        split_mask = large_grad_mask & (self.get_scaling > dense_threshold)
        clone_mask = large_grad_mask & (self.get_scaling <= dense_threshold)
        # split
        N = n_repeat
        stds = self.get_scaling[split_mask].clip(1e-5).unsqueeze(-1).repeat(N,3)*0.1
        means = torch.zeros_like(stds, device="cuda")
        samples = torch.normal(mean=means, std=stds)
        new_xyz = samples + self._xyz[split_mask].repeat(N, 1)
        new_scaling = self.get_scaling[split_mask].repeat(N) / (0.8*N)
        new_features = self._features[split_mask].repeat(N, 1)
        new_opacity = self._opacity[split_mask].repeat(N, 1)
        # clone
        stds = self.get_scaling[clone_mask].clip(1e-5).unsqueeze(-1).repeat(1,3)*0.01
        means = torch.zeros_like(stds, device="cuda")
        # ic(stds)
        samples = torch.normal(mean=means, std=stds)
        new_xyz = torch.cat([new_xyz, samples+self._xyz[clone_mask]])
        new_scaling = torch.cat([new_scaling, self.get_scaling[clone_mask]])
        new_features = torch.cat([new_features, self._features[clone_mask]])
        new_opacity = torch.cat([new_opacity, self._opacity[clone_mask]])
        # update
        new_points_dict = {
            'xyz': new_xyz,
            'scaling': new_scaling,
            'features': new_features,
            'opacity': new_opacity
        }
        # ic(self.get_xyz.shape[0])
        self.densification_postfix(new_points_dict, optimizer)
        # ic(self.get_xyz.shape[0])
        # remove low opacity points
        low_opacity_mask = self.get_opacity < 0.01
        # prune
        n_split = int(torch.count_nonzero(split_mask==True))
        n_clone = int(torch.count_nonzero(clone_mask==True))
        n_low_opacity = int(torch.count_nonzero(low_opacity_mask==True))
        n_map = int(self.get_xyz.shape[0])
        prune_mask = torch.cat([split_mask, torch.zeros(n_map-split_mask.shape[0], device="cuda", dtype=bool)])
        prune_mask = ~torch.logical_or(prune_mask, low_opacity_mask.squeeze())
        self.prune_points(prune_mask, optimizer)
        n_map = int(self.get_xyz.shape[0])
        ic(n_split,n_clone,n_low_opacity,n_map)
        # reset stats
        self.reset_densification_stats()

    def reset_opacity(self, optimizer):
        print("Resetting opacity")
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity", optimizer)
        self._opacity = optimizable_tensors["opacity"]