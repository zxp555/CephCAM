import torch
from pytorch3d.io import load_objs_as_meshes
from modules.img_utils import img2tensor
from PIL import Image
from typing import Optional
from modules.screen_color import ScreenColor
import numpy as np
import random
# import torch.nn.functional as F

class ImageMask:
    def __init__(self, device, mask_file, random_brightness = None, random_bias_r = None):
        # 提取mask文件中的#00FF00为mask
        green = torch.tensor([0.0, 1.0, 0.0]).to(device)
        mask_png = img2tensor(Image.open(mask_file), device)
        self.alpha = (mask_png == green).all(-1, keepdim=True) * 1.0

        self.random_brightness = random_brightness
        self.random_bias_r = random_bias_r
        self.device = device

    # tex: [M, N, 3]
    def mask(self, tex: torch.Tensor, pos, ignore_eot = False, **_):
        if tex is None:
            return None
        
        if ignore_eot:
            return torch.cat((tex, self.alpha), dim=-1)

        if self.random_brightness is not None:
            a,b = self.random_brightness
            bts = random.uniform(a, b)
        else:
            bts = 1
        
        if self.random_bias_r is not None:
            r = self.random_bias_r
            bias = (torch.rand([3], device=self.device)*2-1)*r
        else:
            bias = 0

        return torch.cat(((tex + bias) * bts, self.alpha), dim=-1)

        

def get_P(z_vector, x_vector=[1, 0, 0]):
    up_v = np.array(z_vector)
    up_v = up_v / np.linalg.norm(up_v)

    zero_v = np.array(x_vector)
    zero_v = zero_v - np.dot(up_v, zero_v) * up_v
    zero_v = zero_v / np.linalg.norm(zero_v)

    right_v = np.cross(up_v, zero_v)

    P = np.array([zero_v, right_v, up_v])
    # print(P)
    return P

# ret phi, theta
def map_P(P, sigma, theta):
    camera_v = np.array([
        np.cos(np.deg2rad(sigma)) * np.cos(np.deg2rad(theta)),
        np.cos(np.deg2rad(sigma)) * np.sin(np.deg2rad(theta)),
        np.sin(np.deg2rad(sigma)),
        ])
    camera_v = P @ camera_v

    phi = np.rad2deg(np.arcsin(camera_v[2]))
    theta = np.rad2deg(np.arctan2(camera_v[1], camera_v[0]))
    return phi.item(), theta.item()

class ScreenMask:
    def __init__(self, device, uv_shape, screens, random_brightness = None, random_bias_r = None):
        self.device = device
        # self.uv_shape = uv_shape
        self.screens = screens

        self.random_brightness = random_brightness
        self.random_bias_r = random_bias_r

        Ps = []
        mlps = []
        alpha = torch.zeros([*uv_shape, 1], device=self.device)

        for x1, y1, x2, y2, z_vec, weight in self.screens:
            alpha[y1:y2, x1:x2, :] = 1
            Ps.append(get_P(z_vec))
            mlps.append(ScreenColor(device, weight=torch.load(weight, map_location=device) if weight is not None else None))

        self.Ps = Ps
        self.mlps = mlps
        self.alpha = alpha

    # tex: [M, N, 3]
    def mask(self, tex: torch.Tensor, pos, ignore_eot = False, **_):
        if tex is None:
            return None
        
        bts = torch.ones(tex.shape, device=self.device)
        bias = torch.zeros(tex.shape, device=self.device)
        new_tex = torch.zeros(tex.shape).to(tex.device)
        
        if ignore_eot:
            for i, (x1, y1, x2, y2, _, _) in enumerate(self.screens):
                new_tex[y1:y2, x1:x2, :] = tex[y1:y2, x1:x2, :]
            return torch.cat((new_tex, self.alpha), dim=-1)
        
        if self.random_brightness is not None:
            a,b = self.random_brightness
            for i, (x1, y1, x2, y2, _, _) in enumerate(self.screens):
                bts[y1:y2, x1:x2, :] = random.uniform(a, b)
        
        if self.random_bias_r is not None:
            r = self.random_bias_r
            for i, (x1, y1, x2, y2, _, _) in enumerate(self.screens):
                bias[y1:y2, x1:x2, :] = (torch.rand([3], device=self.device) * 2 - 1) * r

        distance, sigma, theta, roll = pos
        for i, (x1, y1, x2, y2, _, _) in enumerate(self.screens):
            p, t = map_P(self.Ps[i], sigma, theta)
            # print(sigma, theta, p, t)
            new_tex[y1:y2, x1:x2, :] = self.mlps[i].show(tex[y1:y2, x1:x2, :], p, t)

        return torch.cat(((new_tex + bias) * bts, self.alpha), dim=-1)

class CarModel:
    def __init__(self, device, obj_path):
        # 加载点阵
        mesh = load_objs_as_meshes([obj_path], device=device)
        verts = mesh.verts_packed()

        # 缩放
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))
        self.mesh = mesh

        # 原始纹理 [1, M, N, 3]
        self.ori_tex = mesh.textures._maps_padded
        print("Tex shape: ", self.ori_tex.shape)

        self.uv_shape = self.ori_tex.shape[1:3]

    # if tex is None, reset to original texture map
    # tex: [M, N, 4]
    def to_mesh(self, tex: Optional[torch.Tensor]):
        if tex is None:
            self.mesh.textures._maps_padded = self.ori_tex
        else:
            tex = tex.clamp(0.0, 1.0)                       # [M, N, 4]
            rgb = tex[..., :3]                              # [M, N, 3]
            alpha = tex[..., 3:]                            # [M, N, 1]
            tex = rgb * alpha + self.ori_tex * (1 - alpha)  # [1, M, N, 3]
            self.mesh.textures._maps_padded = tex           # [1, M, N, 3]
        return self.mesh
    
    def get_texture(self):
        return self.mesh.textures._maps_padded