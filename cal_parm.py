"""
cal paramaters and flops
coder: Ziqing Ma   time: 2024/09/21
"""
from model import GATmodel
from args_g import args_n
import torch
opt = args_n()
from thop import profile


model = GATmodel(opt.ms_channel, opt.hs_channel, opt.ratio, opt.w_size, opt.stride, opt.dropout, opt.alpha,
opt.ms_feature, opt.hs_feature, opt.neigh).cuda()
input1 = torch.randn(1, 3, 64, 64).cuda()
input2 = torch.randn(1, 31, 8, 8).cuda()
Flops, params = profile(model, inputs=(input2, input1))
print('Number of parameter: %f M' % (params / 1e6))
print('Number of FLOPs: %f M' % (Flops / 1e9))
