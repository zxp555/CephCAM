import torch
from typing import List, Tuple

import sys
path = '/home/zhuxiaopei_srt/data3/env_attack/yolov9'
if path not in sys.path:
    sys.path.append(path)
    print(sys.path)
path = '/home/zhuxiaopei_srt/data3/env_attack'
if path not in sys.path:
    sys.path.append(path)
    print(sys.path)

from yolov9.models.common import DetectMultiBackend
from utils.general import non_max_suppression

class AdapterYolov9:
    def __init__(self, device, img_size, weight):
        # 加载模型
        self.device = device
        self.img_size = img_size
        self.model = DetectMultiBackend(weight, 
                                   device=device, data='../yolov9/data/coco.yaml')
    
    # -> lcls, lobj
    def cpt_loss(self, img: torch.Tensor, target_class: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.model(img) # pred[0]: [1, 84(xyxy, class*80), 8400]
        pred2: torch.Tensor = non_max_suppression(pred) # [1, t, 6(xyxy, conf, class)] 

        scores: torch.Tensor = pred[0][0] # [84(xyxy, class*80), 8400]
        boxes: torch.Tensor = pred2[0] # [t, 6(xyxy, conf, class)]

        lcls = torch.tensor(0.0).to(self.device)
        lobj = torch.tensor(0.0).to(self.device)
        for cls in target_class:
            lcls += scores[cls + 4, :].sum()
            lobj += boxes[:, -2][boxes[:, -1] == cls].sum()

        return lcls, lobj
    
    def run(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.model(img) # pred[0]: [1, 84(xyxy, class*80), 8400]
        # print(pred)
        pred = pred[0]
        pred2: torch.Tensor = non_max_suppression(pred) # [1, t, 6(xyxy, conf, class)] 

        det_bboxes = pred2[0][:, :5]
        det_labels = pred2[0][:, 5]
        det_labels = det_labels.detach().to(torch.int64)
        return det_bboxes, det_labels
