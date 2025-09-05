import cv2
import numpy as np
from select_points import select_points
import os
import matplotlib.pyplot as plt

# 指定一个文件夹，将其中每一个文件做定点操作
# 'Q'键重绘，'C'键确认

# 输出到同名文件.npz
# "spc": 单分量采样数
# "colors": [N, 3], 标准值
# "dataset": [K, N, 3], 实测值
# "angle": [[phi, theta]], 角度值


path = './pic/left'
phi_0_cnt = 0


# 定义单位圆的数据点
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

# 生成若干随机散点
# x=[]
# y=[]
angles=[]

conf = [
    # (20, 6, 0),
    # (40, 6, 0.5),
    # (65, 4, 0),
    # (90, 1, 0),
    # (50, 1, 0.15),

    (20, 6, 0),
    (30, 6, 0.5),
    (50, 4, 0.5),
    (65, 4, 0),
    (90, 1, 0),
    (40, 1, 0.05),
]

for i in range(phi_0_cnt):
    theta = 360/phi_0_cnt*(i+0.5)
    angles.append([0, theta])
    # x.append(np.cos(np.deg2rad(phi))*np.cos(np.deg2rad(theta)))
    # y.append(np.cos(np.deg2rad(phi))*np.sin(np.deg2rad(theta)))


for phi, cnt, bias in conf:
    for i in range(cnt):
        theta = 360/cnt*(i+bias)
        angles.append([phi, theta])
        # x.append(np.cos(np.deg2rad(phi))*np.cos(np.deg2rad(theta)))
        # y.append(np.cos(np.deg2rad(phi))*np.sin(np.deg2rad(theta)))

print(angles)

# plt.plot(x_circle, y_circle)
# plt.scatter(x, y, color='red')

# plt.gca().set_aspect('equal', adjustable='box')

# plt.show()


# 采样常数

sample_size = 600
sinv = 0.15
sample_pixel = [
    (0.5, 0.5),
    (0.5 + sinv, 0.5 + sinv),
    (0.5 + sinv, 0.5 - sinv),
    (0.5 - sinv, 0.5 + sinv),
    (0.5 - sinv, 0.5 - sinv),
]

npz = np.load('./std_colors.npz')
# repeat = npz["repeat"]
grid_size = npz["grid_size"]
shuffle = npz["shuffle"]

def pick_colors(input_image, src_pts):
    # 透视变换
    dst_pts = np.float32([[0, 0], [sample_size, 0], [0, sample_size], [sample_size, sample_size]])
    M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
    rec_image = cv2.warpPerspective(input_image, M, (sample_size, sample_size))

    # 颜色取样
    picked = np.zeros((grid_size * grid_size, 3), dtype=np.uint8)
    sample_image = rec_image.copy()

    block_size = sample_size / grid_size
    for i in range(grid_size * grid_size):
        x = i % grid_size
        y = i // grid_size
        
        sr, sg, sb = 0.0, 0.0, 0.0
        for (dx, dy) in sample_pixel:
            px = int((x + dx) * block_size)
            py = int((y + dy) * block_size)

            b, g, r = rec_image[py, px]
            sample_image[py, px] = 255 - b, 255 - g, 255 - r
            sr += r
            sg += g
            sb += b

        picked[i] = (
            int(sr / len(sample_pixel)), 
            int(sg / len(sample_pixel)), 
            int(sb / len(sample_pixel)),
        )

    picked = picked[shuffle]
    # picked.resize((grid_size * grid_size // repeat, repeat, 3))
    # print(picked.shape)
    # picked = np.average(picked, axis=1).astype(np.uint8)
    # print(picked.shape)

    return picked, sample_image

# 如果确认返回True
def show_sampled(sampled) -> bool:
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Image', sampled)
    cv2.waitKey(10)  # 等待短暂时间以确保窗口最大化
    
    while True:
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord('c'):
            cv2.destroyAllWindows()
            return True
        if (key & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            return False


# A. 模拟图像测试
# input_image = cv2.imread('./pic/wrapped.png')
# src_pts = [[80, 20], [920, 200], [220, 750], [800, 600]]
# picked, sample_image = pick_colors

# 读取图像
# input_image = cv2.imread('./pic/wrapped.png')
# src_pts = np.float32([[80, 20], [920, 200], [220, 750], [800, 600]])

# input_image = cv2.imread('./pic/t0.jpg')
# # 去除摩尔纹
# input_image = cv2.GaussianBlur(input_image, (5, 5), 0)
# src_pts = select_points(input_image)

# print(src_pts)

# angles = []
pickeds = []

filenames = os.listdir(path)
filenames.sort()
if len(angles) != len(filenames) + phi_0_cnt:
    raise "Num error"
    pass
    
for _ in range(phi_0_cnt):
    pickeds.append(np.zeros((grid_size * grid_size, 3), dtype=np.uint8))

import pickle
from PIL import Image
cachepath = f'{path}.c'
os.makedirs(cachepath, exist_ok = True)

for filename in filenames:
    # angle = [int(filename.split('.')[1]), int(filename.split('.')[2])]
    # angles.append(angle)

    while True:
        input_image = cv2.imread(f'{path}/{filename}')
        # 去除摩尔纹
        input_image = cv2.GaussianBlur(input_image, (5, 5), 0)

        points_cache = f'{cachepath}/{filename}.pkl'
        if os.path.isfile(points_cache):
            with open(points_cache, 'rb') as f:
                src_pts = pickle.load(f)
            picked, sample_image = pick_colors(input_image, src_pts)
        else:
            src_pts = select_points(input_image)
            if src_pts is None:
                continue

            picked, sample_image = pick_colors(input_image, src_pts)
            if show_sampled(sample_image):
                with open(points_cache, 'wb') as f:
                    pickle.dump(src_pts, f)
                cv2.imwrite(f'{cachepath}/{filename}', sample_image)
            else:
                continue
        
        pickeds.append(picked)
        break

np.savez(f'{path}{phi_0_cnt}', 
        spc=npz["spc"],
        colors=npz["colors"],
        dataset=np.array(pickeds),
        angle=np.array(angles),
        )

'''
(20, 6, 0),
(30, 6, 0.5),
(50, 4, 0.5),
(65, 4, 0),
(90, 1, 0),
(40, 1, 0.05),
<22>
20 0.0
20 60.0
20 120.0
20 180.0
20 240.0
20 300.0
30 30.0
30 90.0
30 150.0
30 210.0
30 270.0
30 330.0
50 45.0
50 135.0
50 225.0
50 315.0
65 0.0
65 90.0
65 180.0
65 270.0
90 0.0
40 18.0


(20, 6, 0),
(40, 6, 0.5),
(65, 4, 0),
(90, 1, 0),
(50, 1, 0.15),
<18>
20 0.0
20 60.0
20 120.0
20 180.0
20 240.0
20 300.0
40 30.0
40 90.0
40 150.0
40 210.0
40 270.0
40 330.0
65 0.0
65 90.0
65 180.0
65 270.0
90 0.0
50 54.0
'''