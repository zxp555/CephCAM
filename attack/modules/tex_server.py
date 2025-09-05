import cv2
from PIL import Image
import os
import numpy as np

class ImageToVideo:
    def __init__(self, file: str, fps: int):
        self.file = file
        self.fps = fps
        
        # 获取文件的目录并创建它
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        
        # 初始化视频写入器，留出位置来记录视频的参数
        self.out = None
        self.is_initialized = False

        self.frame_cnt = 0
    
    def new_image(self, img: Image.Image):
        # 确保图像分辨率为640x640
        # assert(img.size == (640, 640))
        # 将图像转换为NumPy数组并将颜色空间转换为BGR
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 如果视频写入器未初始化，初始化它
        if not self.is_initialized:
            height, width, layers = img_array.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择视频编码格式
            self.out = cv2.VideoWriter(self.file, fourcc, self.fps, (width, height))
            self.is_initialized = True
        
        # 写入图像到视频文件
        self.out.write(img_array)

        self.frame_cnt+=1

    def save(self):
        # 针对内存泄露，释放视频写入器
        if self.out is not None:
            self.out.release()
            print(f"Video saved to {self.file}, {self.frame_cnt} frames")
            self.out = None  # 重置输出对象
        else:
            print("没有图像写入，无法保存视频。")

class TexServer:
    def __init__(self, path: str, sample_per):
        self.path = path
        self.sample_per = sample_per
        self.tex_id = 0
        self.video_recorder = ImageToVideo(f"{self.path}record.mp4", 30)

    def new_tex(self, tex: Image.Image):
        if self.tex_id % self.sample_per == 0:
            self.video_recorder.new_image(tex)

        if self.tex_id <= 500:
            save = self.tex_id % 20 == 0
        elif self.tex_id <= 3000:
            save = self.tex_id % 120 == 0
        else:
            save = self.tex_id % 800 == 0
        if save:
            tex.save(f"{self.path}tex_{self.tex_id}.png")

        self.tex_id += 1

    def save_video(self):
        self.video_recorder.save()



