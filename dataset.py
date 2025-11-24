'''
read data from prodata
Ziqing Ma
2024/07/07
'''

import torch.utils.data as udata
from datapro import *

class Dataset(udata.Dataset):
    def __init__(self, opt, img_path, txt_path):
        super(Dataset, self).__init__()
        self.respon_path = opt.respon_path
        self.txt_path = txt_path
        self.img_path = img_path
        self.patch_size = opt.patch_size
        self.num_img = opt.num_img
        self.sigma = opt.sigma
        self.k = opt.k
        self.ratio = opt.ratio
        self.data = read_img(self.respon_path, self.txt_path, self.img_path, self.patch_size, 
            self.num_img, self.sigma, self.k, self.ratio)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hrhs = self.data[index][0].float()
        hrms = self.data[index][1].float()
        lrhs = self.data[index][2].float()
        return hrhs, hrms, lrhs
