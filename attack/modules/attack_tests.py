import torch
import random
from modules.logger import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

class AttackTester:
    def __init__(self, dataloader: DataLoader, net, logger: Logger, 
                 confidence_threshold, bound_size, target_class, img_size, sample_cnt):
        self.dataloader = dataloader
        self.net = net
        self.logger = logger
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.bound_size = bound_size
        self.target_class = target_class
        self.test_size = len(self.dataloader)
        self.sample_cnt = sample_cnt

    def test(self, name):
        # 输出若干样例
        tex = self.dataloader.dataset.get_texture()
        if tex is not None:
            self.logger.save_graph(f'{name}.png', tex)

        if self.sample_cnt > 0:
            for i in random.sample(range(self.test_size), self.sample_cnt):
                self.logger.save_graph(f'Ex_{name}/S{i}.png', self.dataloader.dataset[i])

        # 测试
        conf_cnt = len(self.confidence_threshold)
        succ = [0 for _ in range(conf_cnt)]
        pbar = tqdm(self.dataloader)
        for i, (total_img) in enumerate(pbar):
            conf = self.net.detect_obj(total_img, self.target_class, self.bound_size)
            for j in range(conf_cnt):
                if conf < self.confidence_threshold[j]:
                    succ[j] += 1
            pbar.set_description(f'Succ: {succ}')
        
        asrs = []
        for suc in succ:
            rate = suc / self.test_size * 100
            asrs.append(rate)
            
        self.logger.pprint(**{f'Test {name}': asrs})
        return asrs

class MultiTester:
    def __init__(self, dataloader: DataLoader, nets, **v):
        self.dets = [(name, AttackTester(dataloader, net, **v)) for name, net in nets.items()]
        
    def test(self, name):
        for dname, det in self.dets:
            nn = f'{dname}@{name}'
            print("Testing:", nn)
            det.test(nn)