import torch
import torch.nn as nn
from math import cos, sin, radians

# 定义MLP模型
# [r, g, b, x, y] -> [r', g', b']
class ScreenColorMLP(nn.Module):
    def __init__(self):
        super(ScreenColorMLP, self).__init__()
        
        hidden_size = 30

        self.fci = nn.Linear(3 + len(ScreenColorMLP.pos(0, 0)), hidden_size)
        self.relui = nn.ReLU()
        # self.fcm = nn.Linear(h1, h2)
        # self.relum = nn.Sigmoid()
        self.fco = nn.Linear(hidden_size, 3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fci(x)
        out = self.relui(out)
        # out = self.fcm(out)
        # out = self.relum(out)
        out = self.fco(out)
        out = self.sigmoid(out)
        return out
    
    def pos(phi, theta):
        return [
            # phi,
            # theta,

            # cos(radians(phi)),
            # 1 - sin(radians(phi)),
            1 - phi / 90,

            cos(radians(theta)),
            sin(radians(theta)),

            # (1-phi/90) * cos(radians(theta)),
            # (1-phi/90) * sin(radians(theta)),

            # cos(radians(phi)) * cos(radians(theta)),
            # cos(radians(phi)) * sin(radians(theta)),
        ]
    
    def param_cnt(self):
        return sum(p.numel() for p in self.parameters())
    
class ScreenColor:
    def __init__(self, device, weight):
        self.device = device
        if weight is None:
            self.model = None
            return
        
        model = ScreenColorMLP().to(device)
        model.load_state_dict(weight)
        # model = model.to(torch.float64)
        
        model.eval()
        self.model = model

    # img: [M, N, 3], 0~1
    def show(self, img: torch.Tensor, phi, theta) -> torch.Tensor:
        if self.model is None:
            return img
        angle = torch.tensor(ScreenColorMLP.pos(phi, theta), device=self.device)
        angle_expanded = angle.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        angle_expanded = angle_expanded.expand(img.shape[0], img.shape[1], -1)  # [M, N, 3]

        img_expanded = torch.cat((img, angle_expanded), dim=-1) 
        predicted = self.model(img_expanded)

        return predicted

    def test_image(self, phi, theta):
        import matplotlib.pyplot as plt
        from img_utils import tensor2img, pal_img

        img = pal_img().to(self.device)
        predicted = self.show(img, phi, theta)

        plt.subplot(1, 2, 1)
        plt.imshow(tensor2img(img))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(tensor2img(predicted))
        plt.axis('off')

        plt.show()

    def test_pos(self, color):
        import numpy as np

        graph = [[ScreenColorMLP.pos(p, t) for t in range(-179, 181)] for p in range(-45, 135)]
        graph = np.array(graph)

        color = np.array(color).reshape([1,1,3]).repeat(180,axis=0).repeat(360,axis=1)

        graph = np.concatenate((color, graph), axis=-1)

        graph=torch.tensor(graph).to(self.device).to(torch.float32)

        import matplotlib.pyplot as plt
        from img_utils import tensor2img

        graph = self.model(graph)

        plt.imshow(tensor2img(graph))
        plt.axis('off')

        plt.show()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(0)
        print(device)
        torch.cuda.set_device(device)
    else:
        raise

    model = ScreenColor(device, weight=torch.load('./models/front0.pth'))
    model.test_image(30, 0)

    # model.test_pos((211/255, 125/255, 58/255))

    colors = [   (187, 158, 123),
    (211, 125, 58),
    (81, 81, 80),
    (176, 213, 236),
    (235, 237, 231),
    (124, 159, 186),
    (218, 159, 90),
    (153, 92, 55),
    (223, 181, 120),
    (179, 122, 69),
    (227, 204, 158),
    (156, 120, 86),
    (165, 187, 200),
    (146, 70, 33),
    (61, 46, 40),
    (105, 117, 131)] 

    for r, g, b in colors:
        model.test_pos((r/255, g/255, b/255))