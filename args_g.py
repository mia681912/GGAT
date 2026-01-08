import argparse


def args_n():
    parser = argparse.ArgumentParser(description="GGATtrain")
    parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=600, help="Number of training epochs")
    parser.add_argument("--ckpt_step", type=int, default=100, help="Number of save epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--outf", type=str, default="Trained/CAVE", help='path of save file')
    parser.add_argument("--k", type=int, default=5, help='size of deblurred kernel')
    parser.add_argument("--sigma", type=int, default=2, help='size of variance')
    parser.add_argument("--ratio", type=int, default=8, help='resample ratio')
    parser.add_argument("--patch_size", type=int, default=64, help='size of patch')
    parser.add_argument("--num_img", type=int, default=13, help='size of patch num every image')
    parser.add_argument("--val_path", type=str, default='', help='val data root')
    parser.add_argument("--test_path", type=str, default='', help='test data root')
    parser.add_argument("--train_path", type=str, default='', help='train data root')
    parser.add_argument("--val_txt", type=str, default='', help='val text path')
    parser.add_argument("--train_txt", type=str, default='', help='train text path')
    parser.add_argument("--respon_path", type=str, default='', help='reaponse function mat')
    parser.add_argument("--cudanum", type=str, default='1', help='cuda number')
    # model parameter
    parser.add_argument("--ms_channel", type=int, default=3, help='channel for HRMS image')
    parser.add_argument("--hs_channel", type=int, default=31, help='channel for LRHS image')
    parser.add_argument("--dropout", type=float, default=0.6, help='dropout parameter')
    parser.add_argument("--alpha", type=float, default=0.2, help='parameter of nonlinear function')
    parser.add_argument("--hs_feature", type=int, default=64, help='channels for HSI')
    parser.add_argument("--ms_feature", type=int, default=64, help='channel for MSI')
    parser.add_argument("--neigh", type=int, default=8, help='neighbors')
    parser.add_argument("--w_size", type=int, default=16, help='window size')
    parser.add_argument("--stride", type=int, default=8, help='stride')
    opt = parser.parse_args()

    return opt
