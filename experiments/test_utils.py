import torch
import numpy as np
from rkhs_splatting.utils.camera_utils import to_viewpoint_camera
from rkhs_splatting.utils.point_utils import PointCloud

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

def create_camera(H,W,fx,fy,c2w):
    intrinsic = np.eye(4)
    intrinsic[0,0] = fx
    intrinsic[1,1] = fy
    intrinsic[0,2] = W/2
    intrinsic[1,2] = H/2
    intrinsic = intrinsic.reshape(-1)
    c2w = np.array(c2w).reshape(-1)
    camera_data = np.array([H,W, *intrinsic, *c2w], dtype=np.float32)
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