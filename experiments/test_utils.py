import torch
import numpy as np
from rkhs_splatting.utils.camera_utils import to_viewpoint_camera
from rkhs_splatting.utils.point_utils import PointCloud
from rkhs_splatting.utils.data_utils import read_all
from rkhs_splatting.utils.point_utils import get_point_clouds
from icecream import ic
import cv2

def create_pc(data):
    pc_coords = data[:,:3]
    pc_rgbas = data[:,3:]
    pc_channels = dict(
        R = pc_rgbas[:,0],
        G = pc_rgbas[:,1],
        B = pc_rgbas[:,2],
        A = pc_rgbas[:,3]
    )
    pc = PointCloud(pc_coords, pc_channels)
    return pc

def create_camera(fx,fy,cx,cy,c2w):
    intrinsic = np.eye(4)
    intrinsic[0,0] = fx
    intrinsic[1,1] = fy
    intrinsic[0,2] = cx
    intrinsic[1,2] = cy
    intrinsic = intrinsic.reshape(-1)
    c2w = np.array(c2w).reshape(-1)
    camera_data = np.array([cy*2, cx*2, *intrinsic, *c2w], dtype=np.float32)
    return torch.tensor(camera_data).cuda()

def create_dataset(pc, camera_data, model, renderer):
    model.create_from_pcd(pc, initial_scaling=0.5)
    render_output = renderer(
        camera_data[0],
        model.get_xyz,
        model.get_opacity,
        model.get_scaling,
        model.get_features
    )
    return render_output

    # alpha = render_output['alpha']
    # alpha_nobg = torch.where(alpha<1./255., 1, alpha)
    # render_output['depth'] = render_output['depth']/alpha
    # return dict(
    #     camera=camera_data,
    #     rgb=render_output['render'].unsqueeze(0).detach(),
    #     depth=render_output['depth'].squeeze().unsqueeze(0).detach(),
    #     alpha=render_output['alpha'].squeeze().unsqueeze(0).detach(),
    #     # alpha_nobg=alpha_nobg.squeeze().unsqueeze(0).detach()
    # )

def create_random_pc(n, mu=0, sigma=1, alpha=1, rgba=None, shape='none'):
    # coords
    if shape=='none':
        pc_coords = mu + (np.random.rand(n,3)-0.5) * sigma
    elif shape=='sphere_surface':
        pc_coords = np.random.randn(n,3)
        pc_coords = pc_coords / np.linalg.norm(pc_coords, axis=1)[:,None]
        pc_coords = mu + pc_coords * sigma
    elif shape=='sphere':
        pc_coords = np.random.randn(n,3)
        pc_coords = pc_coords / np.linalg.norm(pc_coords, axis=1)[:,None].max()
        pc_coords = mu + pc_coords * sigma
    elif shape=='cube':
        pc_coords = np.random.rand(n,3)
        pc_coords = mu + (pc_coords-0.5) * sigma
    elif shape=='cube_surface':
        pc_coords = np.random.rand(n,3)
        pc_coords = mu + (pc_coords-0.5) * sigma
        face_indices = np.random.choice([0, 1, 2], size=n)
        sign_indices = np.random.choice([1, -1], size=n)
        for i in [0,1,2]:
            pc_coords[face_indices==i, i] = sign_indices[face_indices==i] * sigma/2
    elif shape=='line':
        pc_coords = np.random.rand(n,3)
        pc_coords = mu + (pc_coords-0.5) * sigma
        pc_coords[:,1:] = 0

    # colors
    if rgba is not None:
        pc_rgbas = np.ones((n,4)) * rgba
    else:
        pc_rgbas = np.random.rand(n,4)
    pc_rgbas[:,3] = alpha
    pc_channels = dict(
        R = pc_rgbas[:,0],
        G = pc_rgbas[:,1],
        B = pc_rgbas[:,2],
        A = pc_rgbas[:,3]
    )
    pc = PointCloud(pc_coords, pc_channels)
    return pc

def load_sample_dataset(folder, frame_ranges, resize_factor=0.5):
    train_pcs = []
    cameras = []
    data = read_all(folder, resize_factor=resize_factor)
    data = {k: v.cuda() for k, v in data.items()}
    N = frame_ranges[1] - frame_ranges[0]
    for key,value in data.items():
        data[key] = value[frame_ranges[0]:frame_ranges[1]]
    for i in range(N):
        camera = to_viewpoint_camera(data['camera'][i])
        train_pc = get_point_clouds(camera, data['depth'][i].unsqueeze(0), data['alpha'][i].unsqueeze(0), data['rgb'][i].unsqueeze(0))
        train_pcs.append(train_pc)
        cameras.append(camera)
    return train_pcs, cameras

def load_custom_dataset(dataset, frame_ranges, resize_factor=1):
    train_pcs = []
    cameras = []
    dataset.load_ground_truth()
    for i in range(*frame_ranges):
        dataset.set_curr_index(i)
        rgb, depth = dataset.read_current_rgbd()
        rgb = rgb[:,:,::-1]/255
        depth = depth[:,:,np.newaxis]
        alpha = np.where(depth<100, 1., 0.)
        W, H = dataset.image_size
        new_size = (int(W*resize_factor), int(H*resize_factor))
        cv2.resize(alpha, new_size, interpolation=cv2.INTER_CUBIC)
        rgb, depth, alpha = [cv2.resize(x, new_size, interpolation=cv2.INTER_CUBIC) for x in [rgb, depth, alpha]]
        rgb, depth, alpha = [torch.tensor(x).squeeze().cuda() for x in [rgb, depth, alpha]]
        c2w = dataset.read_current_ground_truth()
        camera_intrinsics = [x*resize_factor for x in dataset.camera]
        camera_data = create_camera(*camera_intrinsics, c2w)
        camera = to_viewpoint_camera(camera_data)
        train_pc = get_point_clouds(camera, depth.unsqueeze(0), alpha.unsqueeze(0), rgb.unsqueeze(0))
        train_pcs.append(train_pc)
        cameras.append(camera)
    return train_pcs, cameras