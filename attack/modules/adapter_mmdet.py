import os
import torch
import torchvision
from torchvision.ops import RoIAlign, RoIPool
import numpy as np
from mmcv.parallel import scatter
from typing import List, Tuple

import sys
path = '/home/zhuxiaopei_srt/data3/env_attack'
if path not in sys.path:
    sys.path.append(path)
    print(sys.path)

from mmdet.apis import init_detector  
from mmdet.datasets.pipelines import Compose

from modules.img_utils import tensor2img

class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], 
                                             std=[58.395, 57.12, 57.375])])
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], 
        #                                      std=[1,1,1])])
        img = transform(results['img'])
        # tensor2img(results['img']).show()
        # tensor2img(img).show()
        # print(img)
        results['img'] = [img]
        return results

# 加载模型
class AdapterMMdet:
    def __init__(self, device, img_size, checkpoint_dir, model_name, config_dir):
        self.device = device
        self.img_size = img_size

        cfg_list = os.listdir(config_dir)
        chk_list = os.listdir(checkpoint_dir)
        for cfg in cfg_list:
            if model_name == 'all' or cfg.split('.')[0] == model_name:
                chk_file = None
                mdl_name = cfg.split('.')[0]
                print(mdl_name)

                # 找到对应的checkpoint_dir
                for chk in chk_list:
                    if chk.split('.')[0].startswith(mdl_name):
                        chk_file = checkpoint_dir + chk
                        break

                cfg_file = config_dir + cfg
                
                config = cfg_file
                checkpoint = chk_file
                self.net = init_detector(config, checkpoint, device=device)
    
    def prepare_data(self, img):
        img = img * 255.0
        data = dict(img=img,
            img_metas=[[{
                        # 'filename': '',  
                        # 'ori_filename': '', 
                        # 'ori_shape': (self.img_size, self.img_size, 3), 
                        'img_shape': (self.img_size, self.img_size, 3),  
                        # 'pad_shape': (800, 800),
                        'scale_factor': np.array([1.0], dtype=np.float32),
                        # 'flip': False, 'flip_direction': None,
                        # 'img_norm_cfg': { 
                        #     'mean': np.array([123.675, 116.28, 103.53], dtype=np.float32),
                        #     'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                        #     'to_rgb': True
                        #     }
                        }]])
        test_pipeline = [LoadImage()]
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(data)
        if next(self.net.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            # raise
            # Use torchvision ops for CPU mode instead
            for m in self.net.modules():
                if isinstance(m, (RoIPool, RoIAlign)):
                    if not m.aligned:
                        # aligned=False is not implemented on CPU
                        # set use_torchvision on-the-fly
                        m.use_torchvision = True
            # warnings.warn('We set use_torchvision=True in CPU mode.')
            # just get the actual data from DataContainer
            # print(data['img_metas'][0])
            # data['img_metas'] = data['img_metas'][0].data
        data['img'][0].data = torch.autograd.Variable(data['img'][0].data).float()
        return data

    # -> lcls, lobj
    def cpt_loss(self, img: torch.Tensor, target_class: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.prepare_data(img)
        results, cls_score, bbox_pred, det_bboxes, det_labels, = self.net(return_loss=False, rescale=True, **data)
        
        cls_score = cls_score[0]
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        lcls = torch.tensor(0.0).to(self.device)
        lobj = torch.tensor(0.0).to(self.device)
        for cls in target_class:
            lcls += torch.softmax(cls_score, dim=1)[:, cls].sum()/cls_score.shape[0]
            obj = det_bboxes[det_labels == cls]
            if obj.shape[0]:
                lobj += obj[:,-1].sum()
            else:
                lobj += 0
        
        return lcls, lobj
    
    def run(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.prepare_data(img)
        r=self.net(return_loss=False, rescale=False, **data)
        results, cls_score, bbox_pred, det_bboxes, det_labels, = r
        return det_bboxes[0], det_labels[0]
