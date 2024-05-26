import torch
import numpy as np

from rkhs_splatting.trainer import Trainer
import rkhs_splatting.utils as utils
import rkhs_splatting.utils.loss_utils as loss_utils
from rkhs_splatting.utils.camera_utils import to_viewpoint_camera, parse_camera
from rkhs_splatting.utils.point_utils import get_point_clouds

import contextlib
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
from pytorch_memlab import LineProfiler
from torch.profiler import profile, ProfilerActivity


USE_GPU_PYTORCH = True
USE_PROFILE = False

class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.input_model = kwargs.get('input_model')
        # self.input_model.set_scaling(self.model.get_scaling)
        self.gauss_render = kwargs.get('renderer')
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        # create a file self.results_folder / f'eval.csv'
        with open(self.results_folder / 'eval.csv', 'w') as f:
            f.write('iter,loss,total,l1,ssim,depth,psnr\n')
        self.writer = kwargs.get('writer', True)
        if self.writer:
            self.tensorboard_writer = SummaryWriter(log_dir=self.results_folder)
        self.use_input_frames = kwargs.get('use_input_frames')
        self.input_frames = kwargs.get('input_frames')
        self.use_rkhs_rgb = kwargs.get('use_rkhs_rgb')
        self.use_rkhs_geo = kwargs.get('use_rkhs_geo')
        self.min_scale = kwargs.get('min_scale', 0.010)
        self.radii_multiplier = kwargs.get('radii_multiplier', 5)
        self.tile_size = kwargs.get('tile_size', 64)
    
    def on_train_step(self):
        ### debug
        # if self.step==2:
        #     quit()

        ### load input frame
        if not self.use_input_frames:
            # load training data
            ind = np.random.choice(len(self.data['camera']))
            camera_data = self.data['camera'][ind]
            camera = to_viewpoint_camera(camera_data)
            rgb = self.data['rgb'][ind]
            depth = self.data['depth'][ind]
            alpha = self.data['alpha'][ind]
            mask = (self.data['alpha'][ind] > 0.5)
            # render input frame
            points = get_point_clouds(self.data['camera'][ind].unsqueeze(0), depth.unsqueeze(0), alpha.unsqueeze(0), rgb.unsqueeze(0))
            self.input_model.set_scaling(self.model.get_scaling)
            self.input_model.create_from_pcd(points, initial_scaling=self.model.get_scaling)
            input_frame = self.gauss_render(
                camera,
                self.input_model.get_xyz,
                self.input_model.get_opacity,
                self.input_model.get_scaling,
                self.input_model.get_features,
                mode='train',
                radii_multiplier=self.radii_multiplier,
                tile_size=self.tile_size
            )
        else:
            ind = np.random.choice(len(self.input_frames))
            input_frame = self.input_frames[ind]
            camera = input_frame['camera']
            rgb = input_frame['render'].detach()
            depth = input_frame['depth'].detach()[..., 0]
            mask = (input_frame['alpha'][..., 0] < 0.5).detach()

        ### profiling tools
        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        ### render current frame
        with prof:
            min_scaling = torch.scalar_tensor(self.min_scale, device="cuda")
            if self.model.get_scaling < min_scaling:
                self.model.set_scaling(min_scaling)
            out = self.gauss_render(
                camera,
                self.model.get_xyz,
                self.model.get_opacity,
                self.model.get_scaling,
                self.model.get_features,
                point_ids=self.model.get_ids,
                radii_multiplier=self.radii_multiplier,
                tile_size=self.tile_size
            )

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))


        ### calc rkhs loss
        rkhs_loss, inner_product_tiles = loss_utils.rkhs_global_scale_loss(out['tiles'], input_frame['tiles'], rgb, self.model.get_scaling, use_geometry=self.use_rkhs_geo, use_rgb=self.use_rkhs_rgb)

        ### remove points with small inner product
        scores = loss_utils.check_rkhs_loss(self.model.get_xyz.shape[0], out['tiles']['id'], inner_product_tiles)
        mask = scores > 0.1
        self.model.add_count(mask)
        if self.step % 100 == 0:
            mask = self.model.get_count>0
            self.model.filter_points(mask)

        ### calc loss
        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)
        # rkhs_loss_total = rkhs_loss[0] + rkhs_loss[1] - 2*rkhs_loss[2]
        rkhs_loss_total = rkhs_loss[0]-2*rkhs_loss[2]
        # rkhs_loss_total = -2*rkhs_loss[2]
        total_loss = rkhs_loss_total
        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr}

        with open(self.results_folder / 'eval.csv', 'a') as f:
            f.write(f'{self.step},{total_loss},{l1_loss},{ssim_loss},{depth_loss},{psnr}\n')

        if self.writer:
            self.tensorboard_writer.add_scalar('loss/total', total_loss, self.step)
            self.tensorboard_writer.add_scalar('loss/rkhs', rkhs_loss_total, self.step)
            self.tensorboard_writer.add_scalar('loss/l1', l1_loss, self.step)
            self.tensorboard_writer.add_scalar('loss/ssim', ssim_loss, self.step)
            self.tensorboard_writer.add_scalar('loss/depth', depth_loss, self.step)
            self.tensorboard_writer.add_scalar('loss/psnr', psnr, self.step)
            self.tensorboard_writer.add_scalar('rkhs/local_map', rkhs_loss[0], self.step)
            self.tensorboard_writer.add_scalar('rkhs/train_frame', rkhs_loss[1], self.step)
            self.tensorboard_writer.add_scalar('rkhs/inner_product', rkhs_loss[2], self.step)
            self.tensorboard_writer.add_scalar('params/scaling', self.model.get_scaling, self.step)
            # self.tensorboard_writer.add_graph(self.gauss_render, [camera_data, self.model.get_xyz, self.model.get_opacity, self.model.get_scaling, self.model.get_features])
            # self.tensorboard_writer.add_image('rgb/render', out['render'], self.step)

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        if not self.use_input_frames:
            ind = np.random.choice(len(self.data['camera']))
            camera_data = self.data['camera'][ind]
            rgb = self.data['rgb'][ind].detach().cpu().numpy()
            depth = self.data['depth'][ind].detach().cpu().numpy()
            # mask = (self.data['alpha'][ind] < 0.5).detach().cpu().numpy()
            # depth[mask] = 0 # set depth for empty area
            camera = to_viewpoint_camera(camera_data)
        else:
            ind = np.random.choice(len(self.input_frames))
            input_frame = self.input_frames[ind]
            camera = input_frame['camera']
            rgb = input_frame['render'].detach().cpu().numpy()
            depth = input_frame['depth'].detach().cpu().numpy()[..., 0]
            # mask = (input_frame['alpha'] < 0.5).detach().cpu().numpy()

        out = self.gauss_render(
            camera,
            self.model.get_xyz,
            self.model.get_opacity,
            self.model.get_scaling,
            self.model.get_features,
            radii_multiplier=self.radii_multiplier,
            tile_size=self.tile_size
        )
        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = depth / depth.max()
        depth = plt.get_cmap('Greys')(depth)[..., :3]

        # draw grid on rgb_pd
        for i in range(0, rgb_pd.shape[1], self.tile_size):
            rgb_pd[:, i] = 0.5
        for i in range(0, rgb_pd.shape[0], self.tile_size):
            rgb_pd[i] = 0.5
        for i in range(0, rgb_pd.shape[1], self.tile_size):
            rgb[:, i] = 0
        for i in range(0, rgb_pd.shape[0], self.tile_size):
            rgb[i] = 0

        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)
        utils.imwrite(str(self.results_folder / f'image-latest.png'), image)

        if self.step == 0:
            utils.imwrite(str(self.results_folder / f'image-gt-rgbd.png'), image[:,:256])
            utils.imwrite(str(self.results_folder / f'image-initial-rgbd.png'), image[:,256:])
        utils.imwrite(str(self.results_folder / f'image-latest-rgbd.png'), image[:,256:])
