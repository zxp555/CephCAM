import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from screen_color import ScreenColorMLP, ScreenColor

# 使用19张版本，3平均

# 超参数
num_epochs = 50000
learning_rate = 0.5
# num_epochs = 20000
# learning_rate = 1
# num_epochs = 100
# learning_rate = 0.001

sets=['top4','front4','left4','top0','front0','left0']
# sets=['top4','front4','left4']
# sets=['front0']

# GPU
if torch.cuda.is_available():
    device = torch.device(0)
    print(device)
    torch.cuda.set_device(device)
else:
    raise

def data_from_npz(filename):
    # 生成训练数据
    npz=np.load(filename)
    c=npz['spc']
    std_colors=npz['colors']        # [N, 3]
    org_data=npz['dataset']         # [K, N, 3]
    angles=npz['angle']             # [K, 2]

    # n=15
    # org_data=org_data[:n]
    # angles=angles[:n]

    c3 = c ** 3
    N = std_colors.shape[0]
    K = angles.shape[0]

    # 角度
    s_pos = [ScreenColorMLP.pos(p, t) for p, t in angles]
    # print(angles)
    # print(s_pos)
    s_pos = np.array(s_pos)
    s_pos = s_pos.reshape((K, 1, s_pos.shape[-1])).repeat(N, axis=1)
    print(s_pos.shape)    # [K, N, 2]
    # raise

    s_colors = (std_colors / 255.0).reshape((1, N, 3)).repeat(K, axis=0)
    print(s_colors.shape)    # [K, N, 3]

    data = np.concatenate((s_colors, s_pos), axis=-1)
    # data = torch.tensor(data, device=device)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    print(data.shape)

    def flatten(t):
        return t.reshape((t.shape[0]*t.shape[1],t.shape[2]))

    # [K*c^3, 5]
    dataset_x = flatten(data[:,:c3,:])
    # [K*R, 5], R = N-c^3
    valset_x = flatten(data[:,c3:,:])


    # data = torch.tensor(org_data / 255.0, device=device)
    data = torch.tensor(org_data / 255.0, dtype=torch.float32, device=device)
    # [K*c^3, 3]
    dataset_y = flatten(data[:,:c3,:])
    # [K*R, 3]
    valset_y = flatten(data[:,c3:,:])


    # ont set -> val

    # [K*R+c^3, 5]
    valset_x = torch.cat((valset_x, dataset_x[-c3:,:]), dim=0)
    # [(K-1)*c^3, 5]
    dataset_x = dataset_x[:-c3,:]
    # [K*R+c^3, 3]
    valset_y = torch.cat((valset_y, dataset_y[-c3:,:]), dim=0)
    # [(K-1)*c^3, 3]
    dataset_y = dataset_y[:-c3,:]

    return dataset_x, dataset_y, valset_x, valset_y

# def data_from_func(func):
#     dataset_x = []
#     dataset_y = []
#     valset_x = []
#     valset_y = []

#     points = []
#     while len(points) < 20:
#         x = np.random.uniform(-1, 1)
#         y = np.random.uniform(-1, 1)
#         if x**2 + y**2 <= 1:
#             points.append((x, y))
#     print(points)

#     ext_colors = [np.random.randint(0, 256, 3) for _ in range(100)]

#     spc = 7
#     print(np.linspace(0, 255, spc))

#     for x, y in points[1:]:
#         for r in np.linspace(0, 255, spc):
#             for g in np.linspace(0, 255, spc):
#                 for b in np.linspace(0, 255, spc):
#                     v = [r / 255.0, g / 255.0, b / 255.0, x, y]
#                     dataset_x.append(v)
#                     dataset_y.append(func(*v))
#         for r, g, b in ext_colors:
#             v = [r / 255.0, g / 255.0, b / 255.0, x, y]
#             valset_x.append(v)
#             valset_y.append(func(*v))

#     for x, y in points[:1]:
#         for r in np.linspace(0, 255, spc):
#             for g in np.linspace(0, 255, spc):
#                 for b in np.linspace(0, 255, spc):
#                     v = [r / 255.0, g / 255.0, b / 255.0, x, y]
#                     valset_x.append(v)
#                     valset_y.append(func(*v))
#         for r, g, b in ext_colors:
#             v = [r / 255.0, g / 255.0, b / 255.0, x, y]
#             valset_x.append(v)
#             valset_y.append(func(*v))

#     return (torch.tensor(dataset_x, device=device, dtype=torch.float32), 
#             torch.tensor(dataset_y, device=device, dtype=torch.float32), 
#             torch.tensor(valset_x, device=device, dtype=torch.float32), 
#             torch.tensor(valset_y, device=device, dtype=torch.float32))
    


# 训练数据
# def func_exact(r, g, b, x, y):
#     return [r, g, b]

# def func_less_blue(r, g, b, x, y):
#     return [r, g, b ** 4]

# import colorsys
# def func_pos(r, g, b, x, y):
#     s = (x**2+y**2)**0.5
#     h = (np.arctan2(y, x) / np.pi + 1) / 2
#     r,g,b=colorsys.hsv_to_rgb(h, s, 0.5)
#     return [r,g,b]

logs = {}

def train_from_set(setname):
    print("Training: ", setname)

    # 初始化模型
    model = ScreenColorMLP().to(device)

    print("Param cnt:", model.param_cnt())

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # dataset_x, dataset_y, valset_x, valset_y = data_from_func(func_pos)
    dataset_x, dataset_y, valset_x, valset_y = data_from_npz(f'./pic/{setname}.npz')

    print(dataset_x.shape)
    print(dataset_y.shape)
    print(valset_x.shape)
    print(valset_y.shape)

    train_losses = []
    val_losses = []

    # train_losses.append(criterion(dataset_x[:,:3], dataset_y).detach().cpu().item())
    # val_losses.append(criterion(valset_x[:,:3], valset_y).detach().cpu().item())

    # direct MSE
    direct_loss = criterion(valset_x[:,:3], valset_y).detach().cpu().item()
    direct_loss = (direct_loss * 3) ** 0.5
    print(f'direct MSE Loss: {direct_loss:.4f}')

    # 训练模型
    tq = tqdm(range(num_epochs))
    for epoch in tq:
        # 前向传播
        loss = criterion(model(dataset_x), dataset_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            output = model(valset_x)
            val_loss = criterion(output, valset_y)
        
        train_loss_v = loss.detach().cpu().item()
        val_loss_v = val_loss.detach().cpu().item()

        train_losses.append(train_loss_v)
        val_losses.append(val_loss_v)

        train_loss_v = (train_loss_v * 3) ** 0.5
        val_loss_v = (val_loss_v * 3) ** 0.5

        tq.set_description(f'Loss: {train_loss_v:.4f}, Val loss: {val_loss_v:.4f}')

    print("Final MSE: ", val_loss_v ** 2 / 3)
    torch.save(model.state_dict(), f'./models/{setname}.pth')
    
    logs[setname] = (train_losses, val_losses)

    # with torch.no_grad():
        # scr = ScreenColor(device, model.state_dict())
    #     scr = ScreenColor(device, torch.load(f'./models/{setname}.pth'))
        # scr.test_image(30, 0)
    #     # scr.test_image(0, 20)
    #     # scr.test_image(60, 20)
    #     # scr.test_image(120, 20)
    #     # scr.test_image(0, 30)
    #     # scr.test_image(0, 50)
    #     # scr.test_image(0, 90)
    #     scr.test_pos((54/255, 77/255, 41/255))
    #     # scr.test_image(180, 0)
    #     # scr.test_image(0, 0)
    #     # scr.test_image(-90, 0)
    #     # scr.test_image(90, 0)
    #     # scr.test_image(0, 90)


for setname in sets:
    train_from_set(setname)

import pickle

# 保存数据
with open('log.pkl', 'wb') as file:
    pickle.dump(logs, file)
 
