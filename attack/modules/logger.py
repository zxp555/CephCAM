import os
import zoneinfo
from PIL import Image
from datetime import datetime
import pickle
from pprint import pformat

import torch
from modules.img_utils import tensor2img

class NullLogger:
    def pprint(self, **values):
        # print(**values)    
        pass

    def get_path(self, name: str) -> str:
        return '/dev/null'

    def save_graph(self, name: str, img):
        # if (isinstance(img, torch.Tensor)):
        #     img = tensor2img(img)

        # if (isinstance(img, Image.Image)):
        #     img.show(name)
        # else:
        #     raise
        pass

class Logger:
    def __init__(self, path, name):
        current_time = datetime.now()
        beijing_tz = zoneinfo.ZoneInfo('Asia/Shanghai')
        beijing_time = current_time.astimezone(beijing_tz)
        formatted_time = beijing_time.strftime('%y%m%d_%H.%M.%S')

        log_dir = f'{path}{formatted_time}_{name}/'
        os.makedirs(log_dir, exist_ok=False)

        self.log_dir = log_dir
        self.txt_file = open(f'{log_dir}info.txt', "w")
        self.pkl = f'{log_dir}info.pkl'

        self.value = {}

    def pprint(self, **values):
        for v in values:
            print(v, ':\n  ', pformat(values[v], indent = 4), '', file=self.txt_file, flush=True)
            self.value[v] = values[v]
        with open(self.pkl, 'wb') as f:
            pickle.dump(self.value, f)        

    def get_path(self, name: str) -> str:
        full_path = self.log_dir + name
        os.makedirs('/'.join(full_path.split('/')[:-1]), exist_ok=True)
        return full_path

    def save_graph(self, name: str, img):
        if (isinstance(img, torch.Tensor)):
            img = tensor2img(img)

        if (isinstance(img, Image.Image)):
            img.save(self.get_path(name))

        else:
            raise