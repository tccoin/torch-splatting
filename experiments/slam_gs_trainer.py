import torch
import numpy as np
import open3d as o3d

import sys
sys.path.append('../')
from rkhs_splatting.utils.camera_utils import parse_camera
from rkhs_splatting.rkhs_model import RKHSModel
from rkhs_splatting.rkhs_render import RKHSRenderer
from rkhs_splatting.rkhs_model_global_scale import RKHSModelGlobalScale
from rkhs_splatting.rkhs_render_global_scale import RKHSRendererGlobalScale
from rkhs_splatting.utils.dataloader import TartanAirLoader, TUMLoader
import rkhs_splatting.utils.loss_utils as loss_utils

import datetime
import pathlib
from icecream import ic
from spatialmath import SE3
import plotly.graph_objects as go

from plotly_utils import *
from test_utils import *
from gs_trainer import GSSTrainer

torch.cuda.set_device(0)

#config
def run_trainer(
        init_map,
        cameras,
        data,
        output_folder,
        tile_size=64,
        map_label = 'test',
    ):

    n_train = 1000
    input_initial_scaling = 0.001
    map_initial_scaling = 0.001
    map_minimum_scaling = 1e-5
    radii_multiplier = 3
    scale_trainable = True
    point_model = 'isotropic' # global_scale, isotropic, anisotropic

    # cameras
    if len(cameras)==0:
        camera_intrinsic = [355, 355, 128, 128] # fx, fy, cx, cy
        n_cameras = 1
        delta_deg = 30.0
        camera_c2w_init = SE3.Tz(-10)
        for i in range(n_cameras):
            camera_c2w = SE3.Ry(delta_deg*i / 180 * np.pi)@camera_c2w_init
            camera_data = create_camera(*camera_intrinsic, camera_c2w)
            cameras.append(to_viewpoint_camera(camera_data))

    # render
    if point_model == 'global_scale':
        map_model = RKHSModelGlobalScale(sh_degree=4, debug=False, trainable=True, scale_trainable=scale_trainable)
        map_model.create_from_pcd(init_map, initial_scaling=map_initial_scaling)
        input_model = RKHSModelGlobalScale(sh_degree=4, debug=False, trainable=False)
        renderer = RKHSRendererGlobalScale(white_bkgd=True)
        rkhs_loss_func = loss_utils.rkhs_loss_global_scale
    elif point_model == 'isotropic':
        map_model = RKHSModel(sh_degree=4, debug=False, trainable=True, scale_trainable=scale_trainable)
        map_model.create_from_pcd(init_map, initial_scaling=map_initial_scaling)
        input_model = RKHSModel(sh_degree=4, debug=False, trainable=False)
        renderer = RKHSRenderer(white_bkgd=True)
        rkhs_loss_func = loss_utils.rkhs_loss

        map_model.update_learning_rate({
            'xyz': {
                'lr_init': 1e-2,
                'lr_final': 1e-3,
                'lr_delay_steps': 0,
                'lr_delay_mult': 1,
                'max_steps': 5000
            },
            'features': {
                'lr_init': 3e-3,
                'lr_final': 1e-3,
                'lr_delay_steps': 0,
                'lr_delay_mult': 1,
                'max_steps': 5000
            },
            'opacity': 5e-2,
            'scaling': 5e-3
        })

    trainer = GSSTrainer(
        model=map_model,
        input_model=input_model,
        renderer=renderer,
        data=data,
        use_input_frames=False,
        use_render=False,
        # input_frames=input_frames,
        train_batch_size=1, 
        train_num_steps=n_train,
        i_image=20,
        train_lr=0, #1e-2
        amp=True,
        fp16=False,
        results_folder=output_folder,
        map_label=map_label,
        use_rkhs_rgb=True,
        use_rkhs_geo=True,
        min_scale=map_minimum_scaling,
        radii_multiplier=radii_multiplier,
        tile_size=tile_size,        
        writer=False,
        outlier_threshold=3e-1,
        filtering_interval=10000,
        densification_interval=100,
        opacity_reset_interval=500,
        rkhs_loss_func=rkhs_loss_func
    )

    trainer.on_evaluate_step()
    trainer.train()
    map_model.save_to(output_folder / f'gs.csv')

if __name__ == '__main__':
    # load dataset
    input_source = 'tum' # tartanair, tum
    dataset_path = '/home/junzhe/Projects/data/'
    if input_source=='tartanair':
        dataset = TartanAirLoader(dataset_path+'tartanair/scenes/abandonedfactory/Easy/P001')
        tile_size = 80
    elif input_source=='tum':
        dataset = TUMLoader(dataset_path+'tum/rgbd_dataset_freiburg1_desk')
        tile_size = 80
    N = dataset.get_total_number()
    print(f'Loading dataset: {dataset.dataset_folder}')

    # create output folder
    output_path = '../result/'+input_source #+'_'+'grad_threshold_5e-4' #+'_'+datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    output_folder = pathlib.Path(output_path)

    # load frame and train
    for i in range(0, N, 10):
        print(f'Processing frame: {i}/{N}')
        frame_folder = output_folder
        frame_folder.mkdir(parents=True, exist_ok=True)
        label = f'{i}'
        map_folder = frame_folder / label
        map_folder.mkdir(parents=True, exist_ok=True)
        train_pcs, cameras, data = load_custom_dataset(dataset, [i,i+1,1], 1)

        # initial map
        stacked_pc = train_pcs[0]
        for pc in train_pcs[1:]:
            stacked_pc = stacked_pc+pc
        stacked_pc.save_pcd(map_folder / f'full_pc.pcd', cameras[0])
        # downsample
        init_map = stacked_pc
        voxel_size = 0.01 #tartanair 0.1
        while len(init_map.coords)>15000:
            print(f'Downsampling with voxel size: {voxel_size}')
            downsampled_pc = stacked_pc.voxel_sample(voxel_size)[0]
            voxel_size *= 1.2
            init_map = downsampled_pc
        print(f'Number of points in initial map: {len(init_map.coords)}')
        init_map.save_pcd(map_folder / f'downsampled_pc.pcd', cameras[0])

        # train gs map
        run_trainer(
            init_map,
            cameras,
            data,
            frame_folder,
            tile_size=tile_size,
            map_label=label,
        )
        # clear memory
        del train_pcs, cameras, data, init_map
        torch.cuda.empty_cache()