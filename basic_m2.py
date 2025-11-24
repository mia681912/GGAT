"""
main modules
coder: Ziqing Ma   time: 2024/09/21
"""


import torch
from torch import nn as nn
import torch.nn.functional as F


class SEattention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEattention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        z = x * y.expand_as(x)
        return z


class ECAAttention(nn.Module):
    def __init__(self, channel, kernrl_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernrl_size, padding=(kernrl_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        z = x * y.expand_as(x)
        return z


class PSAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSAttention, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.collect = nn.Conv2d(out_channels, out_channels, 1)
        self.distribute = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv_reduce(x)
        b, c, h, w = x.size()
        x_collect = self.collect(x).view(b, c, -1)
        print(x_collect.shape)
        x_collect = F.softmax(x_collect, dim=-1)
        x_distribute = self.distribute(x).view(b, c, -1)
        x_distribute = F.softmax(x_distribute, dim=1)
        print((x_distribute.permute(0, 2, 1)).shape)
        x_att = torch.bmm(x_collect, x_distribute.permute(0, 2, 1)).view(b, c, h, w)
        return x + x_att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1)
        return self.sigmoid(x1)


class CBEM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBEM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class GraphspaAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphspaAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features, in_features)))
        self.W2 = nn.Parameter(torch.empty(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, lrgt1):
        Wh = torch.matmul(h, self.W1)
        wlrgt = torch.matmul(lrgt1, self.W2)
        B, N, C = h.size()
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention1 = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention1, dim=2)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        gt_prime = torch.matmul(attention, wlrgt)

        if self.concat:
            return F.elu(h_prime), F.elu(gt_prime)
        else:
            return F.elu(h_prime), F.elu(gt_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.in_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.in_features:, :])

        e = Wh1 + torch.transpose(Wh2, 2, 1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
