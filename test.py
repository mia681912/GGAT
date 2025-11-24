# test and save
# Author: ZiQing Ma
# 2024/09/06

import os
import torch
from torch.autograd import Variable
from args_test import *
import time
from model import GATmodel
import scipy.io as sio
from metrics_cpu import *


opt = args_n()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum
model = GATmodel(opt.ms_channel, opt.hs_channel, opt.ratio, opt.w_size, opt.stride, opt.dropout, opt.alpha,
opt.ms_feature, opt.hs_feature, opt.neigh).cuda()

model_folder = opt.model_path
print(model_folder)

save_result = opt.save_path
if not os.path.exists(save_result):
    os.makedirs(save_result)


def test():
    test_set = opt.data_path
    print(torch.cuda.get_device_name(0))
    num_testing = 320
    test_time = 0.
    sz = 64
    output_HRHSI = np.zeros((num_testing, sz, sz, 31))

    # load model
    print(os.path.join(model_folder,'model/{}.pth'.format(opt.epochs)))
    path_checkpoint = os.path.join(model_folder,'model_best/{}.pth'.format(opt.epochs))
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    iteration = 0
    for i in np.arange(5):
        LRHSI_path = os.path.join(test_set, '{}_{}.mat'.format('LR', str(i)))
        HRMSI_path = os.path.join(test_set, '{}_{}.mat'.format('RGB', str(i)))
        LRHSI = sio.loadmat(LRHSI_path)
        HRMSI = sio.loadmat(HRMSI_path)

        LRHSI = LRHSI["LRHSI"]
        HRMSI = HRMSI['RGB']

        LRHSI = np.swapaxes(LRHSI, 1, 2)
        LRHSI = np.swapaxes(LRHSI, 0, 1)

        HRMSI = np.swapaxes(HRMSI, 1, 2)
        HRMSI = np.swapaxes(HRMSI, 0, 1)

        LRHSI = torch.from_numpy(LRHSI).float()
        HRMSI = torch.from_numpy(HRMSI).float()

        LRHSI, HRMSI = Variable(LRHSI), Variable(HRMSI)
        LRHSI, HRMSI = LRHSI.cuda(), HRMSI.cuda()
        LRHSI = torch.unsqueeze(LRHSI, dim=0)
        HRMSI = torch.unsqueeze(HRMSI, dim=0)

        # print(LRHSI.shape)
        size_pat = sz // opt.ratio
        for j in np.arange(8):
            for l in np.arange(8):
                LRHSI1 = LRHSI[:,:,j*size_pat:(j+1)*size_pat,l*size_pat:(l+1)*size_pat]
                HRMSI1 = HRMSI[:,:,j*sz:(j+1)*sz,l*sz:(l+1)*sz]
                with torch.no_grad():
                    time_s = time.time()
                    output = model(LRHSI1, HRMSI1)
                    test_time += time.time() - time_s

                output = output.squeeze(0)

                output_HRHSI[iteration, :, :, :] = output.permute([1, 2, 0]).cpu().detach().numpy()
                iteration += 1
    iter = 0
    i = 0
    psnr1 = []
    sam1  = []
    ssim1  = []
    ergas1 = []
    while iter < 320:
        # print(iter)
        HRHSI_out = np.zeros((512, 512, 31))
        for j in np.arange(8):
            for l in np.arange(8):
                img1 = output_HRHSI[iter, :, :, :]
                HRHSI_out[j * sz:(j + 1) * sz, l * sz:(l + 1) * sz, :] = img1
                iter += 1
        HSI_path = os.path.join(test_set, '{}_{}.mat'.format('HSI', str(i)))
        HSI = sio.loadmat(HSI_path)

        orig = HSI["HSI"]
        print(orig.shape)
        psnr = psnr_cal(np.clip(orig,0.0,1.0), np.clip(HRHSI_out,0.0,1.0))
        sam = sam_cal(orig, HRHSI_out)
        ssim = ssim_cal(orig, HRHSI_out, 1.)
        ergas = ergas_cal(orig, HRHSI_out, 8)
        
        psnr1.append(psnr)
        sam1.append(sam)
        ssim1.append(ssim)
        ergas1.append(ergas)

        with open(os.path.join(save_result, opt.txt_path), 'a+') as fp:
            fp.write('the-{}-image-result:PSNR= {}, SSIM= {}, ERGAS= {}, SAM= {}\n'
                     .format(i, psnr, ssim, ergas, sam))
            fp.write('the-{}-image-result:&{}&{}&{}&{}\n'
                     .format(i, psnr, ssim, ergas, sam))
        fp.close()
        i += 1
        print(i)
        path = os.path.join(save_result, 'out_{}.mat'.format(str(i-1)))
        sio.savemat(path, {'output': HRHSI_out})

    psnr = np.mean(psnr1)
    ssim = np.mean(ssim1)
    sam = np.mean(sam1)
    ergas = np.mean(ergas1)

    with open(os.path.join(save_result, opt.txt_path), 'a+') as fp:
        fp.write('average result:PSNR= {}, SSIM= {}, ERGAS= {}, SAM= {}\n'
                 .format(psnr, ssim, ergas, sam))
        fp.write('average result:&{}&{}&{}&{}\n'
                 .format(psnr, ssim, ergas, sam))
        fp.write('test time: {}'.format(test_time))
    fp.close()
    print('result:PSNR= {}, SSIM= {}, ERGAS= {}, SAM= {}\n'
                 .format(psnr, ssim, ergas, sam))
    print('result:&{}&{}&{}&{}\n'
                .format(psnr, ssim, sam, ergas))


###################################################################
# ------------------- Main Function  -------------------
###################################################################
if __name__ == "__main__":
    test()




