
import sys
path = '/home/zhuxiaopei_srt/data3/env_attack/yolov9'
if path not in sys.path:
    sys.path.append(path)
    print(sys.path)
import torch
from PIL import Image, ImageDraw, ImageFont
from modules.img_utils import img2tensor, tensor2img, interpolate
from modules.coco import coco_classes

import numpy as np
class Detector:
    def __init__(self, device, img_size, type, **config):
        self.device = device
        self.img_size = img_size
        print("loading: ",config)
        if type == 'yolov9':
            from modules.adapter_yolov9 import AdapterYolov9
            self.d = AdapterYolov9(device, img_size, **config)
        elif type == 'mmdet':
            from modules.adapter_mmdet import AdapterMMdet
            self.d = AdapterMMdet(device, img_size, **config)
        elif type == 'd-detr':
            from modules.adapter_d_detr import AdapterDDetr
            self.d = AdapterDDetr(device, img_size, **config)
        elif type == 'detr':
            from modules.adapter_detr import AdapterDetr
            self.d = AdapterDetr(device, img_size, **config)
        elif type == 'det':
            from modules.adapter_mmdet_det import AdapterDetAny
            self.d = AdapterDetAny(device, img_size, **config)
        else:
            # print(config)
            self.d = config['d']
            # raise

    def pre_process(self, img: Image.Image):
        img = img2tensor(img, self.device)
        img = interpolate(img, [self.img_size, self.img_size])
        img = img.permute((2,0,1)).unsqueeze(0)
        return img

    def run(self, tensor: torch.Tensor):
        return self.d.run(tensor)
    
    def cpt_loss(self, img, target):
        return self.d.cpt_loss(img, target)

    def show(self, tensor: torch.Tensor, result, conf = 0.5, target = None, line_width=None, font_size=None):
        det_bboxes, det_labels = result
        det_bboxes = det_bboxes.detach().cpu().numpy()
        det_labels = det_labels.detach().cpu().numpy()
        # print(det_bboxes.shape)
        # print(det_labels.shape)

        image = np.array(tensor2img(tensor))

        from utils.plots import Annotator, colors 
        annotator = Annotator(image, line_width=line_width, pil=True, font_size=font_size,
                              font="/home/zhuxiaopei_srt/data3/env_attack/code2/modules/consola.ttf")

        # draw = ImageDraw.Draw(image)
        for boxx, cls in zip(det_bboxes, det_labels):
            x0, y0, x1, y1, score = boxx
            if score < conf:
                continue
            if target is not None and cls not in target:
                continue
            # x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))  # 转换为整数
            
            c = int(cls)  # integer class
            label = f'{coco_classes[c]} {score:.2f}'
            # color=colors(c, True)
            color=(255,0,0)
            annotator.box_label((x0, y0, x1, y1), label, color=color)

        image = annotator.result()
        # 保存或显示处理后的图像
        # image.save('output_image.jpg')
        image=Image.fromarray(image)
        return image

    def test(self):
        t = self.pre_process(Image.open('/data3/zhuxiaopei_srt/env_attack/tests/detecter.png'))
        self.show(t, self.run(t))
    
    def detect_obj(self, img: torch.Tensor, target_class, bound_size) -> float:
        det_bboxes, det_labels = self.run(img)

        max_score = 0.0
        for j, cid in enumerate(det_labels):
            bbox = det_bboxes[j].detach().cpu().numpy()
            x1, y1, x2, y2, score = bbox
            midx = (x1 + x2) / 2 / self.img_size
            midy = (y1 + y2) / 2 / self.img_size

            if (cid in target_class and 
                -bound_size < midx - 0.5 < bound_size and
                -bound_size < midy - 0.5 < bound_size and
                score > max_score):
                max_score = score
            
        return max_score