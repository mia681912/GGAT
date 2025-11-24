import argparse


def args_n():
    parser = argparse.ArgumentParser(description="GGAT_test")
    parser.add_argument("--model_path", type=str, default='/home/fgg/file/GATformer_fusion/GAT/paper_harvard_nei12/', help="model path")
    parser.add_argument("--data_path", type=str, default='/home/fgg/file/data/Harvard/Harvard/Harvard_test', help="test data path")
    parser.add_argument("--save_path", type=str, default='Test/Harvard/', help="result save path")
    parser.add_argument("--ratio", type=int, default=8, help="down-sampling ratio")
    parser.add_argument("--txt_path", type=str, default='index.txt', help="index save path")
    parser.add_argument("--epochs", type=int, default=600, help="Number of save epochs")
    parser.add_argument("--patch_size", type=int, default=64, help='size of patch')
    parser.add_argument("--cudanum", type=str, default='1', help='cuda number')
    # 模型参数
    parser.add_argument("--ms_channel", type=int, default=3, help='channel for HRMS image')
    parser.add_argument("--hs_channel", type=int, default=31, help='channel for LRHS image')
    parser.add_argument("--dropout", type=float, default=0.6, help='dropout parameter')
    parser.add_argument("--alpha", type=float, default=0.2, help='parameter of nonlinear function')
    parser.add_argument("--hs_feature", type=int, default=64, help='channels for HSI')
    parser.add_argument("--ms_feature", type=int, default=64, help='channel for MSI')
    parser.add_argument("--neigh", type=int, default=12, help='neighbors')
    parser.add_argument("--w_size", type=int, default=16, help='window size')
    parser.add_argument("--stride", type=int, default=8, help='stride')
    opt = parser.parse_args()
    return opt