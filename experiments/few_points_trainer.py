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
        self.input_model.set_scaling(self.model.get_scaling)
        self.gauss_render = kwargs.get('renderer')
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        # create a file self.results_folder / f'eval.csv'
        with open(self.results_folder / 'eval.csv', 'w') as f:
            f.write('iter,loss,total,l1,ssim,depth,psnr\n')
        self.tensorboard_writer = SummaryWriter(log_dir=self.results_folder)
    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera_data = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        alpha = self.data['alpha'][ind]
        mask = (self.data['alpha'][ind] > 0.5)
        # if USE_GPU_PYTORCH:
        #     camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            # min_scaling = torch.scalar_tensor(0.010, device="cuda")
            # if self.model.get_scaling < min_scaling:
            #     self.model.set_scaling(min_scaling)
            out = self.gauss_render(
                camera_data,
                self.model.get_xyz,
                self.model.get_opacity,
                self.model.get_scaling,
                self.model.get_features
            )

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))

        # if self.step==2:
        #     quit()

        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        points = get_point_clouds(self.data['camera'][ind].unsqueeze(0), depth.unsqueeze(0), alpha.unsqueeze(0), rgb.unsqueeze(0))
        self.input_model.set_scaling(self.model.get_scaling)
        self.input_model.create_from_pcd(points, initial_scaling=self.model.get_scaling)
        input_frame = self.gauss_render(
            camera_data,
            self.input_model.get_xyz,
            self.input_model.get_opacity,
            self.input_model.get_scaling,
            self.input_model.get_features,
            mode='train'
        )

        use_geometry = True
        use_rgb = True
        rkhs_loss = loss_utils.rkhs_global_scale_loss(out['tiles'], input_frame['tiles'], rgb, self.model.get_scaling, use_geometry=use_geometry, use_rgb=use_rgb)
        rkhs_loss_total = rkhs_loss[0] + rkhs_loss[1] - 2*rkhs_loss[2]


        total_loss = rkhs_loss_total
        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr}

        with open(self.results_folder / 'eval.csv', 'a') as f:
            f.write(f'{self.step},{total_loss},{l1_loss},{ssim_loss},{depth_loss},{psnr}\n')

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

        ic(total_loss)

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera_data = self.data['camera'][ind]

        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gauss_render(
            camera_data,
            self.model.get_xyz,
            self.model.get_opacity,
            self.model.get_scaling,
            self.model.get_features
        )
        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        mask = (self.data['alpha'][ind] < 0.5).detach().cpu().numpy()
        depth[mask] = 0 # set depth for empty area
        depth = np.concatenate([depth, depth_pd], axis=1)
        # ic(depth[:,:256].min(), depth[:,:256].max())
        # ic(depth[:,256:].min(), depth[:,256:].max())
        # ic(depth[0,0], depth[0, 256])
        depth = depth / depth.max()
        depth = plt.get_cmap('Greys')(depth)[..., :3]

        # draw grid on rgb_pd
        # for i in range(0, rgb_pd.shape[1], 64):
        #     rgb_pd[:, i] = 0
        # for i in range(0, rgb_pd.shape[0], 64):
        #     rgb_pd[i] = 0

        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)
        utils.imwrite(str(self.results_folder / f'image-latest.png'), image)

        if self.step == 0:
            utils.imwrite(str(self.results_folder / f'image-gt-rgbd.png'), image[:,:256])
            utils.imwrite(str(self.results_folder / f'image-initial-rgbd.png'), image[:,256:])
        utils.imwrite(str(self.results_folder / f'image-latest-rgbd.png'), image[:,256:])