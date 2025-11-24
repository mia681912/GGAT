'''
dataset read from .h5 file
Ziqing Ma
2024/07/07
'''

import torch.utils.data as udata
import h5py as h5
import torch


class Dataset(udata.Dataset):
    def __init__(self, opt, img_path):
        super(Dataset, self).__init__()
        self.img_path = img_path
        self.data = h5.File(self.img_path, "r")

    def __len__(self):
        # print(len(self.data['HSI']))
        return len(self.data['HSI'])

    def __getitem__(self, index):
        hrhs = self.data['HSI'][index][:]
        hrms = self.data['RGB'][index][:]
        lrhs = self.data['LRHSI'][index][:]
        ulrhs = self.data['uLRHSI'][index][:]
        hrhs = torch.from_numpy(hrhs).float()
        hrms = torch.from_numpy(hrms).float()
        lrhs = torch.from_numpy(lrhs).float()
        ulrhs = torch.from_numpy(ulrhs).float()

        hrhs = hrhs.permute(2, 0, 1)
        hrms = hrms.permute(2, 0, 1)
        lrhs = lrhs.permute(2, 0, 1)
        ulrhs = ulrhs.permute(2, 0, 1)

        return hrhs, hrms, lrhs, ulrhs
