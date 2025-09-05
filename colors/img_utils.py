from PIL import Image
import numpy as np
import torch

def is_channel(c):
    return c == 3 or c == 4

# Image -> [M, N, 3] RGB
def img2tensor(img: Image.Image, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img).copy()[...,:3]).to(device) / 255.0

# [M, N] RGB -> Image
# [c, M, N] RGB -> Image
# [M, N, c] RGB -> Image
# [1, c, M, N] RGB -> Image
# [1, M, N, c] RGB -> Image
# c = 3 or 4
def tensor2img(tensor: torch.Tensor) -> Image.Image:
    if len(tensor.shape) != 3:
        if len(tensor.shape) == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        else:
            print(tensor.shape)
            raise "Shape error."
    # -> [c, M, N] / [M, N, c]

    if is_channel(tensor.shape[0]) and not is_channel(tensor.shape[2]):
        tensor = tensor.permute((1, 2, 0))
    # -> [M, N, c]
        
    tensor = torch.clamp(tensor, 0.001, 0.999)
    tensor = tensor.detach().cpu().numpy() * 255
    return Image.fromarray(np.uint8(tensor))

# generate pure color tensor [M, N, 3]
def gen_pure_color_texmap(shape, device, color_RGB = (0, 0, 0)):
    tex = torch.zeros(shape).to(device)
    R, G, B = color_RGB
    tex[:, :, 0] = R
    tex[:, :, 1] = G
    tex[:, :, 2] = B
    return tex

def loss_smooth(img) -> torch.Tensor:
    s1 = torch.pow(img[1:, :-1, :] - img[:-1, :-1, :], 2)
    s2 = torch.pow(img[:-1, 1:, :] - img[:-1, :-1, :], 2)
    return torch.sum((s1 + s2)) / (img.numel())

def sum_images(imgs):
    small_size = 160
    all_image = Image.new('RGB', (small_size * len(imgs), small_size))
    for i, img in enumerate(imgs):
        all_image.paste(img.resize((small_size, small_size)), (small_size * i, 0))
    return all_image

# -> [(R, G, B)], 0~1
def get_dominant_colors(image: Image.Image, color_cnt: int):
    result = image.convert("P", palette=Image.ADAPTIVE, colors=color_cnt)  
    palette = result.getpalette()
    color_counts = sorted(result.getcolors(), reverse=True)
    colors = list()

    for i in range(color_cnt):
        palette_index = color_counts[i][1]
        dominant_color = palette[palette_index * 3 : palette_index * 3 + 3]
        colors.append((dominant_color[0],dominant_color[1],dominant_color[2]))

    return colors

def show_colors(main_colors) -> Image.Image:
    main_color_cnt = len(main_colors)
    W, H = 15, 60

    main_color_show = np.zeros((H, W * main_color_cnt, 3), np.uint8)
    for i in range(main_color_cnt):
        R, G, B = main_colors[i]
        main_color_show[:, W * i:W * (i + 1)] = [R, G, B] 
    return Image.fromarray(main_color_show)

def pal_img() -> torch.Tensor:
    samples_per_color = 9
    grid_size = 27
    rgb_values = torch.linspace(0, 255, samples_per_color).to(torch.uint8)

    pal = torch.zeros((grid_size * grid_size, 3), dtype=torch.uint8)
    for i in range(grid_size * grid_size):
        b = i % samples_per_color
        g = i // samples_per_color % samples_per_color
        r = i // samples_per_color // samples_per_color
        pal[i] = torch.tensor([rgb_values[r], rgb_values[g], rgb_values[b]])

    return pal.reshape((grid_size, grid_size, 3)) / 255.0

class LossColorDistance:
    # img_shape = [M, N, 3]
    # main_colors: (r, g, b)[K] 0~255
    def __init__(self, device, img_shape, main_colors, rg, pown):
        colors = torch.tensor(main_colors, device=device) / 255.0
        colors = colors.unsqueeze(0).unsqueeze(0)
        print(colors.shape)
        colors = colors.repeat([*img_shape[:2], 1, 1])
        print(colors.shape)

        self.colors = colors
        self.K = len(main_colors)
        self.rg = rg
        self.pown = pown / 2

    def loss(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(2).repeat([1, 1, self.K, 1])   # [M, N, K, 3]

        dis2 = torch.sum(torch.pow(img - self.colors, 2), axis=-1) # [M, N, K]
        loss = torch.sum(torch.pow(dis2 / self.rg, -self.pown), axis=-1) # [M, N]
        loss = 1.0 / (loss + 1) # [M, N]
        return torch.mean(loss)