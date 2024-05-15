import pdb
import torch
import torch.nn as nn
import math
from einops import reduce
from icecream import ic
from rkhs_splatting.utils.camera_utils import to_viewpoint_camera

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm



def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]

def build_scale_2d(
    means3d, scale3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (means3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    scale2d_x = (scale3d * focal_x / tz).unsqueeze(-1)
    scale2d_y = (scale3d * focal_y / tz).unsqueeze(-1)
    scale2d = torch.cat((scale2d_x, scale2d_y), dim=1)
    
    return scale2d


def projection_ndc(points, viewmatrix, projmatrix):
    """
    @params points: 3d points
    @params viewmatrix: camera pose
    @params projmatrix: projection matrix
    @return p_proj: projected 2d points
    @return p_view: view space 3d points
    @return in_mask: mask of points whose z > 0.2
    """
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib


class RKHSRendererGlobalScale(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True, **kwargs):
        super(RKHSRendererGlobalScale, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(256), torch.arange(256), indexing='xy'), dim=-1).to('cuda')
    
    def render(self, camera, means3d, scale3d, means2d, scale2d, color, opacity, depths, radii_multiplier, tiles_only=False):
        radii = torch.max(scale2d, dim=-1).values*radii_multiplier
        rect = get_rect(means2d, radii, width=camera.image_width, height=camera.image_height)
        
        if not tiles_only:
            self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
            self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
            self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 64

        h_tile = camera.image_height//TILE_SIZE
        w_tile = camera.image_width//TILE_SIZE

        empty1d = torch.empty(0,1,device='cuda')
        empty2d = torch.empty(0,2,device='cuda')
        empty3d = torch.empty(0,3,device='cuda')
        self.mean2d_tile = {v:{u:empty2d for u in range(w_tile)} for v in range(h_tile)} # h,w,n,2
        self.scale2d_tile = {v:{u:empty2d for u in range(w_tile)} for v in range(h_tile)} # h,w,n,2
        self.mean3d_tile = {v:{u:empty3d for u in range(w_tile)} for v in range(h_tile)} # h,w,n,3
        self.label_tile = {v:{u:[empty3d,empty1d,empty1d] for u in range(w_tile)} for v in range(h_tile)} # h,w,n,5 (3 for RGB, 1 for depth, 1 for opacity)

        for v in range(0, camera.image_height, TILE_SIZE):
            for u in range(0, camera.image_width, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=u), rect[0][..., 1].clip(min=v)
                over_br = rect[1][..., 0].clip(max=u+TILE_SIZE-1), rect[1][..., 1].clip(max=v+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue

                P = in_mask.sum()
                tile_coord = self.pix_coord[v:v+TILE_SIZE, u:u+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2d[in_mask][index]
                sorted_scale2d = scale2d[in_mask][index] # P 2
                sorted_scale2d_squared = torch.max(sorted_scale2d, dim=-1).values**2 # make it 1d
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]
                
                

                self.mean3d_tile[v//TILE_SIZE][u//TILE_SIZE] = means3d[in_mask][index]
                self.mean2d_tile[v//TILE_SIZE][u//TILE_SIZE] = sorted_means2D
                self.scale2d_tile[v//TILE_SIZE][u//TILE_SIZE] = sorted_scale2d
                self.label_tile[v//TILE_SIZE][u//TILE_SIZE] = [sorted_color, sorted_depths, sorted_opacity]

                
                if not tiles_only:
                    dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # B P 2
                    gauss_weight = torch.exp(-0.5 * dx.pow(2).sum(-1) /sorted_scale2d_squared)
                    alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # B P 1
                    T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                    acc_alpha = (alpha * T).sum(dim=1)
                    tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                    tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                    self.render_color[v:v+TILE_SIZE, u:u+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1).clip(min=0, max=1.0)
                    self.render_depth[v:v+TILE_SIZE, u:u+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                    self.render_alpha[v:v+TILE_SIZE, u:u+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)


        tile_data = {
            "mean2d": self.mean2d_tile,
            "scale2d": self.scale2d_tile,
            "mean3d": self.mean3d_tile,
            "label": self.label_tile
        }

        if tiles_only:
            return {
                "tiles": tile_data,
                "camera": camera
            }
        else:
            return {
                "render": self.render_color,
                "depth": self.render_depth,
                "alpha": self.render_alpha,
                "visiility_filter": radii > 0,
                "radii": radii,
                "tiles": tile_data,
                "camera": camera
            }


    def forward(self, camera, means3d, opacity, scale3d, features, mode='render', **kwargs):

        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        with prof("projection"):
            mean_ndc, mean_view, in_mask = projection_ndc(means3d, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            mean_ndc = mean_ndc#[in_mask]
            mean_view = mean_view#[in_mask]
            depths = mean_view[:,2]
        
        with prof("build color"):
            color = features
        
        with prof("scale 2d"):
            scale2d = build_scale_2d(
                means3d=means3d,
                scale3d=scale3d,
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx,
                fov_y=camera.FoVy,
                focal_x=camera.focal_x,
                focal_y=camera.focal_y
            )
            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2d = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

            # ic(scale2d.shape, means2d.shape)

        radii_multiplier = kwargs.get('radii_multiplier', 5)

        with prof("render"):
            if mode=='render':
                rets = self.render(
                    camera = camera, 
                    means3d=means3d,
                    scale3d=scale3d,
                    means2d=means2d,
                    scale2d=scale2d,
                    color=color,
                    opacity=opacity, 
                    depths=depths,
                    radii_multiplier=radii_multiplier
                )
            elif mode=='train':
                rets = self.render(
                    camera = camera, 
                    means3d=means3d,
                    scale3d=scale3d,
                    means2d=means2d,
                    scale2d=scale2d,
                    color=color,
                    opacity=opacity, 
                    depths=depths,
                    tiles_only=True,
                    radii_multiplier=radii_multiplier
                )
        return rets

