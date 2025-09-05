import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw


### 采样

samples_per_color = 8
grid_size = 24

# shuffle_cnt = 3

random.seed(0)

rgb_values = np.linspace(0, 255, samples_per_color).astype(np.uint8)
print(rgb_values)

# 创建一个空白网格
std_colors = np.zeros((grid_size * grid_size, 3), dtype=np.uint8)

for i in range(grid_size * grid_size):
    ii = i
    b = ii % samples_per_color
    ii = ii // samples_per_color
    g = ii % samples_per_color
    ii = ii // samples_per_color
    r = ii
    
    if r < len(rgb_values):
        std_colors[i] = (rgb_values[r], rgb_values[g], rgb_values[b])
    else:
        std_colors[i] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

palette = std_colors
print(palette.shape)

shuffle = list(range(grid_size * grid_size))
random.shuffle(shuffle)
print(shuffle)
inverse_shuffle = np.argsort(shuffle)

palette=palette[shuffle]

np.savez("std_colors", 
         spc=samples_per_color,
         grid_size=grid_size,
         colors=std_colors,
         palette=palette,
         shuffle=inverse_shuffle,
         )


### 生成色板

edge_size = 40
main_size = grid_size * 25
total_size = main_size + 2 * edge_size

# 1. 创建画布
canvas_size = (total_size , total_size)
canvas = Image.new("RGB", canvas_size, "black")
draw = ImageDraw.Draw(canvas)

# 2. 画X形状
draw.line([(0, total_size), (total_size, 0)], fill="white", width=3)
draw.line([(0, 0), (total_size, total_size)], fill="white", width=3)

draw.line([(0, 2 * edge_size), (2 * edge_size, 0)], fill="red", width=7)
draw.line([(0, main_size), (2 * edge_size, total_size)], fill="blue", width=7)
draw.line([(total_size, 2 * edge_size), (main_size, 0)], fill="yellow", width=7)
draw.line([(total_size, main_size), (main_size, total_size - 0)], fill="green", width=7)

# 3. 加载图片并缩放
colors = palette.copy()
print(colors.shape[0])
l=int(colors.shape[0]**0.5)
colors.resize((grid_size, grid_size, 3))

input_image = Image.fromarray(colors)
input_image_resized = input_image.resize((main_size, main_size), Image.NEAREST)

# 将缩放后的图片覆盖到正中间
canvas.paste(input_image_resized, (edge_size, edge_size))

# 保存结果
canvas.save("std_palette.png")


### 生成测试图

import cv2

# 读取输入图像
input_image = cv2.imread('./std_palette.png')

right_edge = main_size + edge_size

# 定义透视变换的顶点
src_pts = np.float32([[edge_size, edge_size], [right_edge, edge_size], [edge_size, right_edge], [right_edge, right_edge]])
dst_pts = np.float32([[80, 20], [920, 200], [220, 750], [800, 600]])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 应用透视变换
transformed_image = cv2.warpPerspective(input_image, M, (1000, 800))

# 计算逆透视变换矩阵
src_pts = np.float32([[0, 0], [main_size, 0], [0, main_size], [main_size, main_size]])
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

# 应用逆透视变换
recovered_image = cv2.warpPerspective(transformed_image, M_inv, (main_size, main_size))

cv2.imwrite('./pic/wrapped.png', transformed_image)

# 显示原图像，透视变换后的图像和还原后的图像
plt.figure(figsize=(15, 5))
plt.subplot(2, 2, 2)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Warped Image')
plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Restored Image')
plt.imshow(cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 1)
plt.imshow(palette[inverse_shuffle].reshape((grid_size, grid_size, 3)))
plt.axis('off')  # 不显示坐标轴

plt.show()
