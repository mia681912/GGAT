"""
calculate psnr, ssim, ergas, sam index
cpu version
coder: Ziqing Ma  time:2024/09/12
"""

import numpy as np


def rmse_cal(img_tgt, img_fus):
    """

    :param img_tgt: [h,w,c]
    :param img_fus: [h,w,c]
    :return: mse
    """

    img_tgt = img_tgt.reshape(img_tgt.shape[2], -1)
    img_fus = img_fus.reshape(img_fus.shape[2], -1)
    rmse = np.sqrt(np.mean((img_tgt - img_fus) ** 2))

    return rmse


def psnr_cal(img_tgt, img_fus):
    """

    :param img_tgt: [c,h,w]
    :param img_fus: [c,h,w]
    :return: PSNR
    """

    # print(img_tgt.shape)
    # print(img_fus.shape)
    img_tgt = img_tgt.reshape(-1, img_tgt.shape[2])  # [c,h*w]
    # print(img_tgt.shape)
    img_fus = img_fus.reshape(-1, img_fus.shape[2])  # [c, h*w]
    # img_tgt = img_tgt * 255.
    # img_fus = img_fus * 255.
    mse = np.mean(np.power(img_tgt - img_fus, 2), axis=0).reshape(1,img_fus.shape[1])   # [c,1] 按通道求均值
    print(np.max(img_fus))
    img_max = np.max(img_fus)
    psnr = 10 * np.log10(img_max ** 2 / mse)
    print(psnr.shape)

    return np.mean(psnr)


def sam_cal(img_tgt, img_fus):
    """

    :param img_tgt: [c,h,w]
    :param img_fus: [c,h,w]
    :return: sam
    """
    # img_fus = np.clip(img_fus, 0, 1.)
    # img_tgt = np.clip(img_tgt, 0, 1.)
    # print()
    img_tgt = img_tgt.reshape([-1, img_tgt.shape[2]])  # [c, h*w]
    img_fus = img_fus.reshape([-1, img_fus.shape[2]])

    A = np.sqrt(np.sum(img_tgt ** 2, axis=1))
    # print(A)
    B = np.sqrt(np.sum(img_fus ** 2, axis=1))
    AB = np.sum(img_tgt * img_fus, axis=1)
    # print(AB.shape)
    # print(np.clip(AB,1e-1,np.inf))
    # print(np.min(np.min(np.clip((A * B),1e-6,np.inf))))
    # print(np.clip(AB,1e-1,np.inf) / np.clip((A * B),1e-1,np.inf))
    sam = AB / np.clip((A * B), 1e-6, np.inf)
    sam = np.clip(sam, a_min=-1, a_max=1)
    # print(np.max(np.max(sam)))
    sam = np.arccos(sam)
    # print(sam)
    sam = np.mean(sam) * 180 / np.pi

    return sam


def ergas_cal(img_tgt, img_fus, scale):
    """

        :param img_tgt: [c,h,w]
        :param img_fus: [c,h,w]
        :param scale: ratio
        :return: ergas
        """

    # print(np.max(img_tgt))
    img_tgt = img_tgt.reshape([-1, img_tgt.shape[2]])
    img_fus = img_fus.reshape([-1, img_fus.shape[2]])

    rmse = np.mean((img_tgt - img_fus) ** 2, axis=0)
    rmse = np.sqrt(rmse)
    mean = np.mean(img_tgt, axis=0)

    ergas = np.mean((rmse / mean) ** 2)
    ergas = 100 / scale * ergas ** 0.5
    # print(ergas)

    return ergas


def ssim_cal(x1, x2, max_v, k1=0.01, k2=0.03):
    """

            :param x1: [c,h,w]
            :param x2: [c,h,w]
            :return: ssim
            """
    c, h, w = x1.shape[2], x1.shape[0], x1.shape[1]
    x1 = x1.reshape([-1, c])
    # print("x1shape", x1.shape)
    x2 = x2.reshape([-1, c])
    # print(np.max(x1))
    u1 = np.mean(x1, axis=0)
    # print(np.mean(x1, axis=1))
    # print("u1shape", u1.shape)
    u2 = np.mean(x2, axis=0)
    Sig1 = np.std(x1, axis=0)
    # print("Sig1shape", Sig1.shape)
    Sig2 = np.std(x2, axis=0)
    sig12 = (np.sum((x1 - u1) * (x2 - u2), axis=0) / (h * w - 1))
    # print("sig12shape", sig12.shape)
    c1, c2 = pow(k1 * max_v, 2), pow(k2 * max_v, 2)
    SSIM = (2 * u1 * u2 + c1) * (2 * sig12 + c2) / ((u1 ** 2 + u2 ** 2 + c1) * (Sig1 ** 2 + Sig2 ** 2 + c2))
    # print(((2 * u1 * u2 + c1) * (2 * sig12 + c2)).shape)
    # print(SSIM.shape)

    return np.mean(SSIM)


