import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from modules.img_utils import img2tensor, interpolate

# 设定种子，使随机数可控
random.seed(0)

def random_crop_and_resize(img_shape):
    # 打开图像
    img_width = img_shape[0]
    img_height = img_shape[1]

    # 确定裁剪区域的边长
    min_side_length = min(img_width, img_height)

    side_length = random.randint(min_side_length//2, min_side_length)

    # 随机选择裁剪区域的左上角坐标
    x1 = random.randint(0, img_width - side_length)
    y1 = random.randint(0, img_height - side_length)

    # 确保裁剪区域不超出原图边界
    x2 = x1 + side_length
    y2 = y1 + side_length

    # 裁剪图像并缩放
    return x1, x2, y1, y2

def train_data_sample(device, cnt, bg_dir, img_size,
                      Distance, Sigma, Theta, Roll):
    res = []

    origin_img = img2tensor(Image.open(bg_dir), device)

    for _ in range(cnt):
        # 距离影响较小，平均即可
        dis_min, dis_max = Distance
        distance = random.uniform(dis_min, dis_max)

        # z服从平均分布
        sigma_min, sigma_max = Sigma
        z = random.uniform(np.sin(np.radians(sigma_min)), 
                              np.sin(np.radians(sigma_max)))
        sigma = np.degrees(np.arcsin(z))

        # 方位角平均分布，(-180, 180]
        theta_min, theta_max = Theta
        theta = random.uniform(theta_min, theta_max)
        theta = theta % 360
        if theta > 180:
            theta -= 360
        if theta < -180:
            theta += 360
            
        roll_min, roll_max = Roll
        roll = random.uniform(roll_min, roll_max)

        # 背景图片
        # background = bg_dir + random.sample(os.listdir(bg_dir), 1)[0]
        # background = Image.open(background)

        # 单背景
        background_config = random_crop_and_resize(origin_img.shape)

        res.append(((distance, sigma, theta, roll), background_config))

    return res, origin_img


# def test_sample(bg_dir):
#     samples = [
#         (10, 0),
#         (10, 30),
#         (10, 60),
#         (10, 90),
#         (10, 120),
#         (10, 150),
#         (10, 180),
#         (10, -150),
#         (10, -120),
#         (10, -90),
#         (10, -60),
#         (10, -30),
#         (30, 0),
#         (30, 45),
#         (30, 90),
#         (30, 135),
#         (30, 180),
#         (30, -135),
#         (30, -90),
#         (30, -45),
#         (60, 0),
#         (60, 90),
#         (60, 180),
#         (60, -90),
#         (90, 0),
#     ]

#     res = []
#     for sigma, theta in samples:
#         # 距离影响较小，平均即可
#         distance = 8

#         roll = 0.0

#         background = bg_dir + os.listdir(bg_dir)[0]
#         background = Image.open(background)

#         res.append(((distance, sigma, theta, roll), background, ))

#     return res

# 数据集
class SimpleBackgroundDataset(Dataset):
    def __init__(self, device, train_set, tex_mask, car, render):
        self.device = device

        res, origin_img = train_set
        self.train_set = res
        self.origin_img = origin_img

        self.tex_mask = tex_mask
        self.car = car
        self.render = render

        self.tex_shape = car.uv_shape

        self.texture = None
    
    def __len__(self) -> int:
        return len(self.train_set)

    def __getitem__(self, index: int) -> torch.Tensor:
        pos, background_config = self.train_set[index]
        
        tex = self.tex_mask.mask(self.texture, pos=pos)
        mesh = self.car.to_mesh(tex)
        imgs = self.render.render(mesh, pos).squeeze(0)

        if imgs.shape[-1] != 640:
            imgs = interpolate(imgs, [640, 640])
        rgb = imgs[..., :3]                                 # [N, N, 3] 
        alpha = imgs[..., 3:]                               # [N, N, 1]

        x1, x2, y1, y2 = background_config
        bg = interpolate(self.origin_img[x1:x2, y1:y2, :], [640, 640])
        img = (rgb * alpha + bg * (1 - alpha)).permute((2, 0, 1))   # [3, N, N]
        return img
        
    def set_texture(self, tex):
        if tex is None:
            self.texture = None
        else:
            self.texture = interpolate(tex, self.tex_shape)

    def get_texture(self):
        pos, _ = self.train_set[0]
        tex = self.tex_mask.mask(self.texture, pos=pos, ignore_eot=True)
        _ = self.car.to_mesh(tex)
        return self.car.get_texture()