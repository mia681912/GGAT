'''
cave dataset
read and make dataset
Ziqing Ma
2024/07/07
'''

import cv2
import scipy.io as sio
import os
import pandas as pd
import numpy as np
import torch


def from_num_tensor(data):
    data = torch.from_numpy(data).permute(2, 0, 1)
    return data


def read_img(respon_path, txtpath, img_path, patch_size, num_img, sigma, k, ratio):
    file1 = pd.read_csv(txtpath).values[:, 0]
    file = []
    for i in range(len(file1)):
        file.append(file1[i])

    num = 0
    data_aug = []
    # spectral response
    respon = sio.loadmat(respon_path)
    respon = respon['P']
    num = 0
    for l in range(len(file1)):
        data = sio.loadmat(os.path.join(img_path, file[l]))
        # print(data)
        # [512, 512, 31]
        # cave 
        # hrhs = data['orig']
        # Harvard
        hrhs = data['gt']
        hrms = np.tensordot(hrhs, respon, axes=(2, 1))

        w, h = hrhs.shape[0], hrhs.shape[1]
        # print([w h])
        stride = (w - patch_size + 1) // num_img  
        # print("=========================")
        # print(stride)
        for i in range(0, w - patch_size + 1, stride):
            # print(i)
            for j in range(0, h - patch_size + 1, stride):
                # print(j)
                rel_hrhs = hrhs[i:i + patch_size, j:j + patch_size, :]
                rel_hrms = hrms[i:i + patch_size, j:j + patch_size, :]
                rel_lrhs = cv2.GaussianBlur(rel_hrhs, (k, k), sigma)
                rel_lrhs = cv2.resize(rel_lrhs, (patch_size // ratio, patch_size // ratio))

                rel_hrhs = from_num_tensor(rel_hrhs)
                rel_hrms = from_num_tensor(rel_hrms)
                rel_lrhs = from_num_tensor(rel_lrhs)

                data_aug.append([rel_hrhs, rel_hrms, rel_lrhs])
                num += 1
                # print(num)
        # print(len(data_aug))
    print("sum patches %d" % num)
    return data_aug


if __name__ == "__main__":
    img_path = 'E:\CLASS\paper\dataset\Original\CAVE\save_cave'
    patch_size = 64
    num_img = 13
    sigma = 2
    k = 5
    ratio = 4
    dataset = read_img(img_path, patch_size, num_img, sigma, k, ratio)
    print(len(dataset))
    print(dataset[0][0].shape)








