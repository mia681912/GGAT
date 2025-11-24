# code for my model for LRHS-LRMS fusion
# Author: ZiQing Ma
# 2024/09/06

import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import Dataset
from model import GATmodel
import numpy as np
import time
from args_g import args_n
from metrics_gpu import *
import random

random_seed = 999999
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

opt = args_n()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cudanum

# load parameters
lr = opt.lr
epochs = opt.epochs
ckpt_step = opt.ckpt_step
batch_size = opt.batchSize

# load model
model = GATmodel(opt.ms_channel, opt.hs_channel, opt.ratio, opt.w_size, opt.stride, opt.dropout, opt.alpha,
                 opt.ms_feature, opt.hs_feature, opt.neigh).cuda()

# loss and optimization setting
ModelLoss = nn.L1Loss().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200,
                                              gamma=0.1)

# file save
model_folder = os.path.join(opt.outf, 'model')
if not os.path.isdir(model_folder):
    os.makedirs(model_folder)

model_folder1 = os.path.join(opt.outf, 'model_best')
if not os.path.isdir(model_folder1):
    os.makedirs(model_folder1)

argsDict = opt.__dict__
with open(os.path.join(opt.outf, 'setting.txt'), 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')
    f.close()


# save model
def save_checkpoint(model, epoch):
    model_out_path = os.path.join(model_folder, "{}.pth".format(epoch))

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_checkpoint1(model, epoch, epoch1):
    model_out_path = os.path.join(model_folder1, "{}.pth".format(epoch1))

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "lr": lr
    }
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    torch.save(checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


###################################################################
# ------------------- Main Train ----------------------------------
###################################################################

def train(training_data_loader, validate_data_loader, start_epoch=0, RESUME=False):
    if RESUME:
        path_checkpoint = os.path.join(model_folder1, "{}.pth".format(200))
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print('Network is Successfully Loaded from %s' % (path_checkpoint))
    best_psnr = 0
    best_sam = 100
    train_time = 0
    best_ergas = 100
    for epoch in range(start_epoch, epochs, 1):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []
        psnr_train, psnr_val = [], []
        ssim_train, ssim_val = [], []
        ergas_train, ergas_val = [], []
        sam_train, sam_val = [], []
        lr_list = []
        # ============Epoch Train=============== #
        model.train()
        for iteration, batch in enumerate(training_data_loader, 1):
            GT, HRMSI, LRHSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            optimizer.zero_grad()
            time_s = time.time()
            output_HRHSI = model(LRHSI, HRMSI)
            Myloss = ModelLoss(output_HRHSI, GT)
            epoch_train_loss.append(Myloss.item())
            time_e = time.time()
            train_time += (time_e - time_s)
            # ===============================calculate index=======================================
            sam = sam_cal(GT, output_HRHSI).item()
            psnr = psnr_cal(GT, output_HRHSI).item()
            ssim = ssim_cal(GT, output_HRHSI, 1.0).item()
            ergas = ergas_cal(GT, output_HRHSI, 8).item()

            psnr_train.append(psnr)
            ssim_train.append(ssim)
            ergas_train.append(ergas)
            sam_train.append(sam)

            Myloss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.6f}  PSNR: {:.2f}  "
                      "SSIM: {:.4f}  ERGAS: {:.4f}  SAM: {:.4f}"
                      .format(epoch, iteration, len(training_data_loader),Myloss.item(), psnr, ssim, ergas, sam))
        lr_list.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()  # update lr
        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        t_psnr = np.nanmean(np.array(psnr_train))
        t_ssim = np.nanmean(np.array(ssim_train))
        t_ergas = np.nanmean(np.array(ergas_train))
        t_sam = np.nanmean(np.array(sam_train))
        if epoch % ckpt_step == 0:
            save_checkpoint(model, epoch)
        print('Epoch: {}/{} training loss: {:.7f} lr: {:.8f}'.format(epochs, epoch, t_loss, optimizer.param_groups[0]['lr']))  # print loss for each epoch
        print('Epoch: {}/{} training psnr: {:.2f}'.format(epochs, epoch, t_psnr))
        print('Epoch: {}/{} training ssim: {:.4f}'.format(epochs, epoch, t_ssim))
        print('Epoch: {}/{} training ergas: {:.4f}'.format(epochs, epoch, t_ergas))
        print('Epoch: {}/{} training sam: {:.4f}'.format(epochs, epoch, t_sam))

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    GT, HRMSI, LRHSI = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                    output_HRHSI = model(LRHSI, HRMSI)
                    Pixelwise_Loss = ModelLoss(output_HRHSI, GT)
                    sam = sam_cal(GT, output_HRHSI).item()
                    psnr = psnr_cal(GT, output_HRHSI).item()
                    ssim = ssim_cal(GT, output_HRHSI, 1.0).item()
                    ergas = ergas_cal(GT, output_HRHSI, 8).item()
                    psnr_val.append(psnr)
                    ssim_val.append(ssim)
                    ergas_val.append(ergas)
                    sam_val.append(sam)
                    Myloss = Pixelwise_Loss
                    epoch_val_loss.append(Myloss.item())  # save all losses into a vector for one epoch

            v_loss = np.nanmean(np.array(epoch_val_loss))
            v_psnr = np.nanmean(np.array(psnr_train))
            v_ssim = np.nanmean(np.array(ssim_train))
            v_ergas = np.nanmean(np.array(ergas_train))
            v_sam = np.nanmean(np.array(sam_train))
            with open(os.path.join(opt.outf, 'valid.txt'), 'a+') as fp:
                fp.write('epoch: {} validate loss: {:.7f} PSNR: {:.2f} SSIM: {:.4f}'
                         'ERGAS: {:.4f}  SAM: {:.4f}\n'.format(epoch, v_loss, v_psnr, v_ssim, v_ergas, v_sam))
            fp.close()

            print("             learning rate:º%f" % (optimizer.param_groups[0]['lr']))
            print('             validate loss: {:.7f}'.format(v_loss))
            print('             validate psnr: {:.2f}'.format(v_psnr))

            if best_psnr < v_psnr and best_sam > v_sam and best_ergas > v_ergas:
                best_psnr = np.mean(psnr_val)
                best_sam = np.mean(sam_val)
                best_ergas = np.mean(ergas_val)
                save_checkpoint1(model, epoch, 600)
                with open(os.path.join(opt.outf, 'train.txt'), 'a+') as fp:
                    fp.write("epochs: {} PSNR: {:.2f} SAM: {:.4f} ergas: {:.4f}\n".format(epoch, best_psnr, best_sam, best_ergas))
                fp.close()

    with open(os.path.join(opt.outf, 'train.txt'), 'a+') as fp:
        fp.write("total train time: {:.4f}\n".format((time_e - time_s)))
    fp.close()


###################################################################
# ------------------- Main Function  -------------------
###################################################################
if __name__ == "__main__":
    print("start")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())
    train_set = Dataset(opt, opt.train_path, opt.train_txt)
    # train_set = Dataset(opt, opt.train_path)
    print("achieve")
    training_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    validate_set = Dataset(opt, opt.val_path, opt.val_txt)
    # validate_set = Dataset(opt, opt.val_path)
    validate_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    train(training_data_loader, validate_data_loader)


