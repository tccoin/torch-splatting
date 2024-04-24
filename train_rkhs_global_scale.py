import torch
import numpy as np
import rkhs_splatting.utils as utils
from rkhs_splatting.trainer import Trainer
import rkhs_splatting.utils.loss_utils as loss_utils
from rkhs_splatting.utils.data_utils import read_all
from rkhs_splatting.utils.camera_utils import to_viewpoint_camera
from rkhs_splatting.utils.point_utils import get_point_clouds, get_point_clouds_tiles
from rkhs_splatting.gauss_model import GaussModelGlobalScale
from rkhs_splatting.gauss_render import GaussRendererGlobalScale
import datetime
import pathlib
from icecream import ic

import contextlib

from pytorch_memlab import LineProfiler
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

USE_GPU_PYTORCH = True
USE_PROFILE = False

class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRendererGlobalScale(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        # create a file self.results_folder / f'eval.csv'
        with open(self.results_folder / 'eval.csv', 'w') as f:
            f.write('iter,loss,total,l1,ssim,depth,psnr\n')
        self.tensorboard_writer = SummaryWriter(log_dir=self.results_folder)


    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]
        alpha = self.data['alpha'][ind]
        mask = (self.data['alpha'][ind] > 0.5)
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            # max_scaling = torch.scalar_tensor(0.02, device="cuda")
            # if self.model.get_scaling > max_scaling:
            #     self.model.set_scaling(max_scaling)
            out = self.gaussRender(pc=self.model, camera=camera)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))



        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        # ic(self.data['camera'][ind].unsqueeze(0).shape)
        # ic(depth.shape)
        # ic(alpha.shape)
        # ic(rgb.shape)

        points = get_point_clouds_tiles(self.data['camera'][ind].unsqueeze(0), depth.unsqueeze(0), alpha.unsqueeze(0), rgb.unsqueeze(0))


        rkhs_loss = loss_utils.rkhs_global_scale_loss(out['tiles'], points, rgb, self.model.get_scaling)


        # total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        total_loss = rkhs_loss[0] + rkhs_loss[1] - 2*rkhs_loss[2]
        # total_loss = -rkhs_loss[2]
        psnr = utils.img2psnr(out['render'], rgb)
        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr}

        with open(self.results_folder / 'eval.csv', 'a') as f:
            f.write(f'{self.step},{total_loss},{l1_loss},{ssim_loss},{depth_loss},{psnr}\n')

        self.tensorboard_writer.add_scalar('loss/total', total_loss, self.step)
        self.tensorboard_writer.add_scalar('loss/rkhs_loss0', rkhs_loss[0], self.step)
        self.tensorboard_writer.add_scalar('loss/rkhs_loss1', rkhs_loss[1], self.step)
        self.tensorboard_writer.add_scalar('loss/rkhs_loss2', rkhs_loss[2], self.step)
        self.tensorboard_writer.add_scalar('loss/l1', l1_loss, self.step)
        self.tensorboard_writer.add_scalar('loss/ssim', ssim_loss, self.step)
        self.tensorboard_writer.add_scalar('loss/depth', depth_loss, self.step)
        self.tensorboard_writer.add_scalar('params/scaling', self.model.get_scaling, self.step)
        
        self.tensorboard_writer.add_scalar('psnr', psnr, self.step)

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)


if __name__ == "__main__":
    device = 'cuda'
    folder = './data/B075X65R3X'
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)

    # ic(data['camera'].shape)
    # ic(data['depth'].shape)
    # ic(data['alpha'].shape)
    # ic(data['rgb'].shape)


    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    raw_points = points.random_sample(2**14)
    # raw_points.write_ply(open('points.ply', 'wb'))

    gaussModel = GaussModelGlobalScale(sh_degree=4, debug=False)
    gaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': True,
    }
    # folder_name = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    folder_name = 'test'
    results_folder = pathlib.Path('result/'+folder_name)
    results_folder.mkdir(parents=True, exist_ok=True)

    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=25000,
        i_image =100,
        train_lr=1e-5,
        amp=False,
        fp16=False,
        results_folder=results_folder,
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()

    open('result/rkhs_global_scale_loss.txt', 'w').write('')

    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=60, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/junzhe/Projects/torch-splatting/result/rkhs_loss_trace'),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    # with LineProfiler(trainer.gaussRender.render) as lp:
    with LineProfiler(trainer.gaussRender.render) as lp:
        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            print('done')
    lp.print_stats(
        stream=open('result/rkhs_global_scale_loss.txt', 'a'),
        columns=('active_bytes.all.allocated', 'active_bytes.all.freed','active_bytes.all.current','active_bytes.all.peak', 'reserved_bytes.all.current')
    )