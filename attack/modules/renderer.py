import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
)

from pytorch3d.transforms import euler_angles_to_matrix

class Renderer:
    def __init__(self, device, img_size, rander_scale, rander_car_at, fov):
        # 渲染设置
        lights = DirectionalLights(device=device, 
                                    ambient_color=((0.7, 0.7, 0.7),),
                                    diffuse_color=((0.4, 0.4, 0.4),),
                                    specular_color=((0.2, 0.2, 0.2),),
                                    direction=((1, 1, -1),))
        raster_settings = RasterizationSettings(image_size=img_size,
                                                blur_radius=0.0, 
                                                faces_per_pixel=1,
                                                bin_size=0)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, lights=lights)
        )
        
        self.rander_scale = rander_scale
        self.rander_car_at = rander_car_at
        self.fov = fov
        self.device = device

    # -> shape=[1,w,h,c=4] val in [0,1] 
    def render(self, mesh, pos):
        distance, sigma, theta, roll_angle = pos
        R, T = look_at_view_transform(dist = distance * self.rander_scale,
                                      elev = sigma,  # 0~180
                                      azim = 180+theta,  # -180~180
                                      at = (self.rander_car_at,),  # (1, 3)
                                      up = ((0, 1, 0),),  # (1, 3)
                                      device=self.device)

        # 计算 roll 旋转矩阵，绕 z 轴旋转
        roll_angle_rad = torch.tensor(roll_angle * (torch.pi / 180.0), device=self.device)  # 将度数转换为弧度
        R_roll = euler_angles_to_matrix(roll_angle_rad.expand(1, 3), "ZYX")  # 绕 z 轴旋转

        # 应用 roll 旋转到 R 矩阵
        R = R @ R_roll
        
        camera = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = camera
        self.renderer.shader.cameras = camera

        imgs_pred = self.renderer(mesh)
        alpha = imgs_pred[..., 3:]
        imgs_pred[..., 3:] = torch.where(alpha == 0, 0, 1)
        return torch.clamp(imgs_pred, 0, 1)