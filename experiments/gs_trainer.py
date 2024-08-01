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
        self.lambda_depth = 0.1 # tartanair 0.05
        # output folder
        self.map_label = kwargs.get('map_label')
        self.map_folder = self.results_folder / self.map_label
        self.map_folder.mkdir(parents=True, exist_ok=True)
        with open(self.map_folder / 'eval.csv', 'w') as f:
            f.write('iter,n_points,total,l1,ssim,depth,psnr,sky\n')
        with open(self.map_folder / 'train.csv', 'w') as f:
            f.write('iter,n_points,total,l1,ssim,depth,psnr,sky\n')
        self.writer = kwargs.get('writer', True)
        if self.writer:
            self.tensorboard_writer = SummaryWriter(log_dir=self.map_folder)
        self.use_input_frames = kwargs.get('use_input_frames')
        self.use_render = kwargs.get('use_render')
        self.input_frames = kwargs.get('input_frames')
        self.use_rkhs_rgb = kwargs.get('use_rkhs_rgb')
        self.use_rkhs_geo = kwargs.get('use_rkhs_geo')
        self.min_scale = kwargs.get('min_scale', 0.010)
        self.radii_multiplier = kwargs.get('radii_multiplier', 5)
        self.tile_size = kwargs.get('tile_size', 64)
        self.outlier_threshold = kwargs.get('outlier_threshold', 0.1)
        self.filtering_interval = kwargs.get('filtering_interval', 50)
        self.opacity_reset_interval = kwargs.get('opacity_reset_interval', 500)
        self.densification_interval = kwargs.get('densification_interval', 50)
        self.rkhs_loss_func = kwargs.get('rkhs_loss_func', loss_utils.rkhs_loss_global_scale)
        # self.fixed_positions = kwargs.get('fixed_positions', False)

    def on_train_step(self):
        # load training data
        ind = np.random.choice(len(self.data['camera']))
        camera_data = self.data['camera'][ind]
        camera = to_viewpoint_camera(camera_data)
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        alpha = self.data['alpha'][ind]
        mask = alpha > 0.5

        ### profiling tools
        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        # if self.fixed_positions:
        #     self.model.get_xyz.requires_grad = False

        ### render current frame
        with prof:
            # self.model.set_scaling(self.model.get_scaling.clip(min=self.min_scale))
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
            self._out = out

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))


        ### calc rkhs loss
        # rkhs_loss, inner_product_tiles = self.rkhs_loss_func(out['tiles'], input_frame['tiles'], rgb, self.model.get_scaling, use_geometry=self.use_rkhs_geo, use_rgb=self.use_rkhs_rgb) 
        # self._inner_product_tiles = inner_product_tiles

        ### calc loss
        loss_file = self.map_folder / 'train.csv'
        sky_loss = 0
        if self.data.get('sky_mask') is not None:
            sky_mask = self.data['sky_mask'][ind]
            rgb[sky_mask.repeat(3, axis=2)] = 1
            sky_loss = loss_utils.l1_loss(out['alpha'][sky_mask], 0)
        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        out_depth = out['depth']
        # out_depth = out['depth'].clip(min=1)
        out_alpha = out['alpha'].clip(min=0.0001)
        out_expected_depth = (out_depth/out_alpha)[..., 0]
        depth_loss = loss_utils.l1_loss(out_expected_depth[mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)
        total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth + 0.1*sky_loss
        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr, 'sky': sky_loss}

        with open(loss_file, 'a') as f:
            n_points = self.model.get_xyz.shape[0]
            f.write(f'{self.step},{n_points},{total_loss},{l1_loss},{ssim_loss},{depth_loss},{psnr},{sky_loss}\n')

        if self.writer:
            self.tensorboard_writer.add_scalar('train_loss/total', total_loss, self.step)
            self.tensorboard_writer.add_scalar('train_loss/l1', l1_loss, self.step)
            self.tensorboard_writer.add_scalar('train_loss/ssim', ssim_loss, self.step)
            self.tensorboard_writer.add_scalar('train_loss/depth', depth_loss, self.step)
            self.tensorboard_writer.add_scalar('train_loss/sky', sky_loss, self.step)
            self.tensorboard_writer.add_scalar('train_loss/psnr', psnr, self.step)

        self.model.set_learning_rate(self.opt, self.step)
        self.opt.zero_grad()

        return total_loss, log_dict
    
    def after_backward_step(self):
        out = self._out
        # inner_product_tiles = self._inner_product_tiles
        ### densify points
        self.model.add_densification_stats()

        ### remove points with small inner product
        # scores = loss_utils.check_rkhs_loss(self.model.get_xyz.shape[0], out['tiles']['id'], inner_product_tiles)
        # count_mask = scores > self.outlier_threshold
        # self.model.add_count(count_mask)
        # if self.step>0 and self.step % self.filtering_interval == 0:
        #     inlier_mask = self.model.get_count>0
        #     self.model.prune_points(inlier_mask, self.opt)
        #     self.model.reset_id_and_count()

        # stop densification and opacities reset
        stop_until = self.train_num_steps*0.8
        step_in_range = self.step>0 and self.step<stop_until

        if step_in_range and self.step % self.densification_interval == 0:
            self.model.densify(self.opt)

        if step_in_range and self.step % self.opacity_reset_interval == 0:
            self.model.reset_opacity(self.opt)
        
        # if self.step==5:
        #     self.model.densify(self.opt)
        #     self.model.reset_opacity(self.opt)

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        if self.step==0:
            # self._evaluation_ind = np.random.choice(len(self.data['camera']))
            self._evaluation_ind = 0
        ind = self._evaluation_ind
        camera_data = self.data['camera'][ind]
        camera = to_viewpoint_camera(camera_data)
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        alpha = self.data['alpha'][ind]
        mask = alpha > 0.5

        out = self.gauss_render(
            camera,
            self.model.get_xyz,
            self.model.get_opacity,
            self.model.get_scaling,
            self.model.get_features,
            radii_multiplier=self.radii_multiplier,
            tile_size=self.tile_size
        )

        ### calc loss
        loss_file = self.map_folder / 'eval.csv'
        sky_loss = 0
        if self.data.get('sky_mask') is not None:
            sky_mask = self.data['sky_mask'][ind]
            rgb[sky_mask.repeat(3, axis=2)] = 1
            sky_loss = loss_utils.l1_loss(out['alpha'][sky_mask], 0)
        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        out_depth = out['depth']
        out_alpha = out['alpha'].clip(min=0.0001)
        out_expected_depth = (out_depth/out_alpha)[..., 0]
        depth_loss = loss_utils.l1_loss(out_expected_depth[mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)
        total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth + 0.1*sky_loss
        psnr = utils.img2psnr(out['render'], rgb)

        with open(loss_file, 'a') as f:
            n_points = self.model.get_xyz.shape[0]
            stats = [self.step, n_points, total_loss, l1_loss, ssim_loss, depth_loss, psnr, sky_loss]
            str_stats = ','.join(['{:.4f}'.format(x) for x in stats])
            f.write(str_stats)

        if self.writer:
            self.tensorboard_writer.add_scalar('eval_loss/total', total_loss, self.step)
            self.tensorboard_writer.add_scalar('eval_loss/l1', l1_loss, self.step)
            self.tensorboard_writer.add_scalar('eval_loss/ssim', ssim_loss, self.step)
            self.tensorboard_writer.add_scalar('eval_loss/depth', depth_loss, self.step)
            self.tensorboard_writer.add_scalar('train_loss/sky', sky_loss, self.step)
            self.tensorboard_writer.add_scalar('eval_loss/psnr', psnr, self.step)

        # save images
        out_rgb = out['render'].detach().cpu().numpy()
        out_depth = out['depth'].clip(min=1)
        out_alpha = out['alpha'].clip(min=0.001)
        out_expected_depth = (out_depth/out_alpha).detach().cpu().numpy()[..., 0]

        # draw grid
        # for i in range(0, out_rgb.shape[1], self.tile_size):
        #     out_rgb[:, i] = 0.5
        # for i in range(0, out_rgb.shape[0], self.tile_size):
        #     out_rgb[i] = 0.5
        # for i in range(0, out_rgb.shape[1], self.tile_size):
        #     rgb[:, i] = 0
        # for i in range(0, out_rgb.shape[0], self.tile_size):
        #     rgb[i] = 0

        depth = depth.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()
        image_depth = np.concatenate([depth, out_expected_depth], axis=1).clip(max=20)
        image_depth = image_depth / image_depth.max()
        image_depth = plt.get_cmap('plasma')(image_depth)[..., :3]
        image_rgb = np.concatenate([rgb, out_rgb], axis=1)
        image = np.concatenate([image_rgb, image_depth], axis=0)

        if self.step==0:
            steps_folder = self.map_folder / 'steps'
            steps_folder.mkdir(parents=True, exist_ok=True)
            utils.imwrite(str(self.map_folder / f'image-initial.png'), image)
            utils.imwrite(str(self.results_folder / f'image-initial.png'), image)
        utils.imwrite(str(self.map_folder / f'steps/image-{self.step}.png'), image)
        utils.imwrite(str(self.map_folder / f'image-latest.png'), image)
        utils.imwrite(str(self.results_folder / f'image-latest.png'), image)
