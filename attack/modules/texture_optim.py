import torch
import torch.nn.functional as F
import numpy

class DirectTexture:
    # uv_shape: [W, H]
    def __init__(self, device, org_shape, init=None):
        if init is None:
            tex_map = torch.rand([*org_shape, 3]).to(device)
        else:
            tex_map = torch.load(init, map_location=device)

        self.tex_map = torch.autograd.Variable(tex_map, requires_grad=True)

    def params(self):
        return [self.tex_map]

    def texture(self):
        return self.tex_map

class GumbleSoftmaxTexture:
    # shape: [W, H]
    # main_colors: [(R, G, B)], 0~1
    def __init__(self, device, org_shape, main_color_cnt, tau, fixed_softmax_g=False, init=None):
        self.device = device
        self.tau = tau
        self.prob_map_shape = [*org_shape, main_color_cnt]

        self.fixed_softmax_g = fixed_softmax_g

        if init is None:
            # V1: 最初的版本
            # prob_map = torch.zeros(self.prob_map_shape, device=device) + 0.5
            # self.seed = -torch.log(-torch.log(torch.rand(self.prob_map_shape, device=self.device)))
            if not self.fixed_softmax_g:
                # V2: 正确的gumbel softmax, 初始值可以改
                prob_map = torch.zeros(self.prob_map_shape, device=device)
            else:
                # V3: 退化的softmax版本, 初始值可以改
                prob_map = self.new_seed()
                # prob_map = -torch.log(torch.rand(self.prob_map_shape, device=self.device))
        else:
            prob_map = torch.load(init, map_location=device)

        self.prob_map = torch.autograd.Variable(prob_map, requires_grad=True)

    def new_seed(self):
        r = torch.rand(self.prob_map_shape, device=self.device)
        # eps = 1e-8
        eps = 0
        return -torch.log(-torch.log(r + eps) + eps)

    def texture(self):
        # V1: 最初的版本
        # color_map = F.softmax((self.prob_map + self.seed) / self.tau, dim=-1)
        if not self.fixed_softmax_g:
            # V2: 正确的gumbel softmax
            color_map = F.softmax((self.prob_map + self.new_seed()) / self.tau, dim=-1)
        else:
            # V3: 退化的softmax版本
            color_map = F.softmax(self.prob_map / self.tau, dim=-1)

        tex = torch.matmul(color_map, self.main_color)
        return tex

    def params(self):
        return [self.prob_map]

    def set_main_colors(self, main_colors):
        self.main_color = torch.tensor(
                numpy.array(main_colors).astype('float32') / 255.0
            ).to(self.device)
        # print(self.main_color)