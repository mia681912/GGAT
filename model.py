"""
GGAT: High-frequency geometry enhanced graph attention network for hyperspectral and multispectral image fusion
coder: Ziqing Ma   time: 2024/09/21
"""

from basic_m2 import *


class SpatialGATtention(nn.Module):
    def __init__(self, ms_channel, out_channel, nfeat, nout, dropout, alpha, w_size, stride, hs_channel, neigh):
        super(SpatialGATtention, self).__init__()
        self.neigh = neigh
        self.w_size = w_size
        self.stride = stride
        self.conv_1 = nn.Sequential(
            nn.Conv2d(ms_channel, out_channel, 3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(hs_channel, out_channel*2, 3, padding=1, bias=False),
            # nn.BatchNorm2d(rank),
            nn.LeakyReLU()
        )
        self.SpaGAT = GraphspaAttentionLayer(nfeat, nout*2, dropout, alpha)
        self.conv2 = nn.Conv2d(out_channel*4, hs_channel, 1)
        self.conv3 = nn.Conv2d(out_channel*2, hs_channel, 1)

    def forward(self, hrms, lrgt1):
        hrms = self.conv_1(hrms)
        lrgt = self.conv_2(lrgt1)
        ksizes = self.w_size
        strides = self.stride
        [b, c, h, w] = hrms.shape
        hrms_matrix = self.extra_tensor_patches(hrms, ksizes, strides)
        # print(hrms_matrix.shape)
        hrms_matrix = hrms_matrix.permute(0, 2, 1)  # [32, 9, 768]
        # print(hrms_matrix.shape)
        # print(lrgt.shape)
        lrgt_matrix = self.extra_tensor_patches(lrgt, ksizes, strides)
        lrgt_matrix = lrgt_matrix.permute(0, 2, 1)  # [32, 9, 1792]
        # print(lrgt_matrix.shape)

        B, N = hrms_matrix.shape[0], hrms_matrix.shape[1]
        adj_ma = torch.zeros([B, N, N]).cuda()  # [8,49,49]
        one_matrix = torch.ones([N, N]).cuda()
        zero_matrix = torch.zeros([N, N]).cuda()
        for i in range(B):
            adj = self.adj_matrix(hrms_matrix[i, :, :])  # [65536,65536]
            adj_ma[i, :, :] = torch.where(adj > 0, one_matrix, zero_matrix)
        lrms, out_gt = self.SpaGAT(hrms_matrix, adj_ma, lrgt_matrix)  # [B,N,C]

        out_gt = out_gt.permute(0, 2, 1).contiguous()  # [B,C,N]
        out_gt = torch.nn.functional.fold(out_gt, (h, w), ksizes, stride=strides)
        out_gt = self.conv2(torch.cat([out_gt, lrgt], 1))

        lrms = lrms.permute(0, 2, 1).contiguous()  # [B,C,N]
        lrms = torch.nn.functional.fold(lrms, (h, w), ksizes, stride=strides)
        out_ms = self.conv3(torch.cat([lrms, hrms], 1))

        return out_gt, out_ms

    def adj_matrix(self, x):
        adj = F.cosine_similarity(x[:, None], x, dim=2)
        topk_values, topk_indices = torch.topk(adj, k=self.neigh + 1, dim=1)  # +1
        adj1 = torch.zeros_like(adj)
        adj1.scatter_(1, topk_indices, 1.0)

        return adj1

    def extra_tensor_patches(self, image, ksizes, strides):
        unfold = torch.nn.Unfold(kernel_size=ksizes, padding=0, stride=strides)
        patches = unfold(image)

        return patches


class SEDE(nn.Module):
    def __init__(self, in_channel, hs_feature):
        super(SEDE, self).__init__()

        self.in_channel = in_channel
        self.hs_feature = hs_feature

        self.ct = ChannelAttention(self.hs_feature // 8, 2)
        self.hs_end = nn.Sequential(
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.BatchNorm2d(self.hs_feature),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature // 2, kernel_size=1),
            # nn.BatchNorm2d(self.hs_feature // 2),
            nn.LeakyReLU(),
            # nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature // 2, self.hs_feature // 4, kernel_size=1),
            # nn.BatchNorm2d(self.hs_feature // 4),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature // 4, self.hs_feature // 8, kernel_size=1),
            # nn.BatchNorm2d(self.hs_feature // 8),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature // 8, self.hs_feature // 8, kernel_size=3, padding=1)
        )

        self.hs_dec = nn.Sequential(
            nn.Conv2d(self.hs_feature // 8, self.hs_feature // 8, kernel_size=3, padding=1),
            # nn.BatchNorm2d(self.hs_feature // 8),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature // 8, self.hs_feature // 4, kernel_size=1),
            # nn.BatchNorm2d(self.hs_feature // 4),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature // 4, self.hs_feature // 2, kernel_size=1),
            # nn.BatchNorm2d(self.hs_feature // 2),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature // 2, self.out_feature, kernel_size=1),
            # nn.BatchNorm2d(self.hs_feature),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_feature, self.out_feature, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.hs_end1(x)
        x = x * self.ct(x)
        x = self.hs_dec(x)

        return x


class FeaFusion(nn.Module):
    def __init__(self, out_channel):
        super(FeaFusion, self).__init__()
        self.hs_feature = out_channel
        self.spa_fusion = nn.Conv2d(2 * self.hs_feature, self.hs_feature, 1)
        self.spe_fusion = nn.Sequential(
            nn.Conv2d(self.hs_feature, self.hs_feature, 1)
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
        )

    def forward(self, fea_spe, fea_spa, fea_gx):
        out = torch.cat([fea_spa, fea_gx], 1)
        out = self.spa_fusion(out)
        out = self.spe_fusion(fea_spe + out)

        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Res_bottle(nn.Module):
    def __init__(self, in_channel):
        super(Res_bottle, self).__init__()

        self.hs_feature = in_channel
        res_conv = nn.Sequential(
            nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, 3, padding=1),
        )
        self.res1 = Residual(res_conv)

    def forward(self, lrhs):
        out = self.res1(lrhs)

        return out


class GMSF(nn.Module):
    def __init__(self, hsi_channels, msi_channels, out_channels):
        super(GMSF, self).__init__()

        self.hsi_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hsi_channels, out_channels, 3,
                          padding=d, dilation=d),
                nn.ReLU()
            ) for d in [1, 3, 5]  # dilation rates
        ])

        self.msi_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(msi_channels, out_channels, 3,
                          padding=d, dilation=d),
                nn.ReLU()
            ) for d in [1, 3, 5]
        ])

        self.compress = nn.Conv2d(6 * out_channels, out_channels, 1)

        self.mss = nn.Conv2d(3 * out_channels, out_channels, 1)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels//2, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )


        self.res_conv1 = nn.Conv2d(hsi_channels, out_channels, 1)
        self.res_conv2 = nn.Conv2d(hsi_channels, out_channels, 1)
        self.res_conv = nn.Conv2d(hsi_channels + msi_channels, out_channels, 1)

    def forward(self, hsi_feat, msi_feat):
        hsi_scales = []
        for conv in self.hsi_convs:
            hsi_scales.append(conv(hsi_feat))

        msi_scales = []
        for conv in self.msi_convs:
            msi_scales.append(conv(msi_feat))

        hsi_all = torch.cat(hsi_scales, dim=1)  # [B, 3*C_out, H, W]
        msi_all = torch.cat(msi_scales, dim=1)  # [B, 3*C_out, H, W]
        combined = torch.cat([hsi_all, msi_all], dim=1)  # [B, 6*C_out, H, W]

        compressed = self.compress(combined)  # [B, C_out, H, W]

        hsi_proj = F.relu(self.res_conv1(hsi_feat))[:, :compressed.size(1)]  # [B,C_out,H,W]
        msi_proj = F.relu(self.res_conv2(msi_feat))[:, :compressed.size(1)]  # [B,C_out,H,W]

        gate_input = torch.cat([hsi_proj, msi_proj], dim=1)  # [B, 2*C_out, H, W]
        gates = self.gate_conv(gate_input)  # [B, 2, H, W]

        fused = gates * compressed + (1 - gates) * msi_proj

        final_out = fused + hsi_feat

        return final_out, self.mss(msi_all)


class SRAttention(nn.Module):
    def __init__(self, in_channel, out_channel, k_size1):
        super(SRAttention, self).__init__()
        layers = []
        for i in range(1):
            layers.append(CBEM(out_channel))
        self.SSRA = nn.Sequential(*layers)
        # out_channel = in_channel
        # k_size1 = 3
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.ms_local = Local_ms(out_channel, out_channel)
        self.conv1 = nn.Conv2d(2 * out_channel, out_channel, 1)

    def forward(self, lrgt):
        lrgt = self.conv2(lrgt)
        # print(lrgt.shape)
        out1 = self.SSRA(lrgt)
        out2 = self.ms_local(out1)
        out = self.conv1(torch.cat([out1, out2], 1))

        return out


class Local_ms(nn.Module):
    def __init__(self, ms_channel, out_channel):
        super(Local_ms, self).__init__()
        k_size4 = 3
        k_size1 = 3
        k_size2 = 3
        # out_channel = 128
        self.conv2 = nn.Conv2d(ms_channel, out_channel, 1)
        self.conv71 = nn.Conv2d(out_channel, out_channel, k_size4, padding=(k_size4 - 1) // 2, bias=False)
        self.conv72 = nn.Conv2d(out_channel, out_channel, k_size4, padding=(k_size4 - 1) // 2, bias=False)
        self.conv51 = nn.Conv2d(out_channel, out_channel, k_size1, padding=(k_size1 - 1) // 2, bias=False)
        self.conv52 = nn.Conv2d(out_channel, out_channel, k_size1, padding=(k_size1 - 1) // 2, bias=False)
        self.conv31 = nn.Conv2d(out_channel, out_channel, k_size2, padding=(k_size2 - 1) // 2, bias=False)
        self.conv32 = nn.Conv2d(out_channel, out_channel, k_size2, padding=(k_size2 - 1) // 2, bias=False)
        # self.conv1 = nn.Conv2d(out_channel, out_channel, 1)
        self.lu1 = nn.LeakyReLU()
        self.lu2 = nn.LeakyReLU()
        self.lu3 = nn.LeakyReLU()
        self.lu4 = nn.LeakyReLU()

    def forward(self, lrhs):
        lrhs = self.conv2(lrhs)
        x1 = self.conv72(self.lu1(self.conv71(lrhs)))
        x2 = self.conv52(self.lu2(self.conv51(x1)))
        x3 = self.conv32(self.lu3(self.conv31(x2)))

        out = x3
        return out


class SpaEmbe(nn.Module):
    def __init__(self, ms_channel, out_channel):
        super(SpaEmbe, self).__init__()
        k_size4 = 3
        k_size1 = 3
        k_size2 = 3
        # out_channel = 128
        self.conv7 = nn.Conv2d(ms_channel, out_channel, k_size4, padding=(k_size4 - 1) // 2, bias=False)
        self.conv5 = nn.Conv2d(ms_channel, out_channel, k_size1, padding=(k_size1 - 1) // 2, bias=False)
        self.conv3 = nn.Conv2d(ms_channel, out_channel, k_size2, padding=(k_size2 - 1) // 2, bias=False)
        # self.conv1 = nn.Conv2d(ms_channel, out_channel, k_size3, bias=False)
        self.lu1 = nn.LeakyReLU()
        self.lu2 = nn.LeakyReLU()
        self.lu3 = nn.LeakyReLU()
        self.lu4 = nn.LeakyReLU()
        self.conv = nn.Conv2d(3 * out_channel, out_channel, 1, bias=False)

    def forward(self, lrhs):
        out7 = self.lu1(self.conv7(lrhs))
        out5 = self.lu2(self.conv5(out7))
        out3 = self.lu3(self.conv3(out5))

        return out3


class CSSA(nn.Module):
    def __init__(self, hs_feature):
        super(CSSA, self).__init__()
        
        self.hs_feature = hs_feature
        self.spa_att = SpatialAttention()
        self.spe_att = ChannelAttention(self.hs_feature)
        self.conv_lrgt = SpaEmbe(self.hs_feature, self.hs_feature)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1, groups=16),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1, groups=8),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1, groups=4),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1)
        )
        self.fusion_att = nn.Sequential(
            nn.Conv2d(int(self.hs_feature), self.hs_feature, 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
        )

        self.fusion_att1 = nn.Sequential(
            nn.Conv2d(int(2 *self.hs_feature), self.hs_feature, 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
        )

        self.fusion_ms = nn.Sequential(
            nn.Conv2d(int(2 * self.hs_feature), self.hs_feature, 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
        )

        self.res = nn.Sequential(
            Res_bottle(self.hs_feature),
            Res_bottle(self.hs_feature)
        )

    def forward(self, hs, ms):
        ms1 = self.conv_lrgt(ms)
        ms_att = self.spa_att(ms1)

        hs1 = self.encoder(hs)
        hs_att = self.spe_att(hs1)

        ms_out1 = ms1 * hs_att

        ms_out = self.fusion_ms(torch.cat([ms1 * ms_att, ms1 * hs_att], 1))

        lrgt1 = hs * hs_att
        lrgt2 = hs * ms_att

        lr_out = self.fusion_att(lrgt1 + lrgt2)
        lrgt2 = self.res(lr_out)

        return lrgt2, ms_out


class GATmodel(nn.Module):
    def __init__(self, ms_inchannel, hs_inchannel, ratio, w_size, stride, dropout, alpha,
                 ms_feature, hs_feature, neigh):
        super(GATmodel, self).__init__()

        self.ms_inchannel = ms_inchannel
        self.hs_inchannel = hs_inchannel
        self.ratio = ratio
        self.w_size = w_size

        self.hs_feature = hs_feature
        self.ms_feature = ms_feature
        self.ms_inchannel1 = self.ms_feature // 8
        self.gmsf1 = GMSF(self.hs_feature, self.ms_feature, self.hs_feature)
        self.gmsf2 = GMSF(self.hs_feature, self.ms_feature, self.hs_feature)
        self.gmsf3 = GMSF(self.hs_feature, self.ms_feature, self.hs_feature)

        self.gt_att = SRAttention(self.hs_feature, self.hs_feature, 3)
        # self.conv_hs = nn.Conv2d(self.hs_inchannel, self.hs_feature, kernel_size=3, padding=1)
        self.conv_ms = nn.Conv2d(self.ms_inchannel, self.ms_feature, kernel_size=3, padding=1)
        self.fusion_up2 = nn.Sequential(
            nn.Conv2d(self.hs_inchannel + self.ms_inchannel, self.hs_feature, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            # nn.LeakyReLU(),
            # nn.Conv2d(self.hs_feature, self.hs_feature * 8 * 8, kernel_size=3, padding=1)
        )

        self.sede1 = SEDE(self.hs_feature, self.hs_feature)
        self.sede2 = SEDE(self.hs_feature, self.hs_feature)


        self.spa_graph1 = SpatialGATtention(self.ms_feature, self.ms_inchannel1,
                                            self.ms_inchannel1 * self.w_size * self.w_size,
                                            self.ms_inchannel1 * self.w_size * self.w_size,
                                            dropout, alpha, w_size, stride, self.hs_feature, neigh)

        self.spa_graph2 = SpatialGATtention(self.ms_feature, self.ms_inchannel1,
                                            self.ms_inchannel1 * self.w_size * self.w_size,
                                            self.ms_inchannel1 * self.w_size * self.w_size,
                                            dropout, alpha, w_size, stride, self.hs_feature, neigh)

        self.fusion = FeaFusion(self.hs_feature)
        self.fusion1 = FeaFusion(self.hs_feature)

        self.fusion2 = nn.Conv2d(int(self.ms_feature + self.hs_feature), self.hs_feature, kernel_size=1)
        # 离散小波变换
        self.fusion3 = nn.Conv2d(int(self.hs_feature), self.hs_feature, 1)
        self.fusion_gt = nn.Conv2d(int(2 * self.hs_feature), self.hs_feature, kernel_size=1)
        self.fusion_att = nn.Conv2d(int(self.hs_feature), self.hs_feature, 1)
        self.fusion_ms = nn.Sequential(
            nn.Conv2d(int(2 * self.hs_feature), self.hs_feature, 1),
        )
        self.fusion_ms1 = nn.Sequential(
            nn.Conv2d(int(self.hs_feature), self.hs_feature, 1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.hs_feature, self.hs_feature, kernel_size=3, padding=1),
        )
        self.res1 = nn.Sequential(
            Res_bottle(self.hs_feature),
            Res_bottle(self.hs_feature)
        )
        self.res2 = nn.Sequential(
            Res_bottle(self.hs_feature),
            Res_bottle(self.hs_feature)
        )

        self.cssa1 = CSSA(self.hs_feature)
        self.cssa2 = CSSA(self.hs_feature)

        self.gt_out = nn.Conv2d(int(self.hs_feature), self.hs_inchannel, kernel_size=3, padding=1)

    def forward(self, lrhs, hrms):
        lrout = torch.nn.functional.interpolate(lrhs, size=(hrms.shape[2], hrms.shape[3]), mode='bilinear')
        lrgt1 = self.fusion_up2(torch.cat([lrout, hrms], 1))
        hrms = self.conv_ms(hrms)

        lrgt1, hrms = self.gmsf1(lrgt1, hrms)

        lrgt_gap = self.sede1(lrgt1)

        grad = abs(self.Grad_x(hrms)) + abs(self.Grad_y(hrms))
        lrgt_grad = abs(self.Grad_x(lrgt1)) + abs(self.Grad_y(lrgt1))

        msgt_gap1, out_ms1 = self.spa_graph1(hrms, lrgt1)
        msgt_gapx, out_ms2 = self.spa_graph2(grad, lrgt_grad)

        hrms = hrms + self.fusion_ms(torch.cat([out_ms1, out_ms2], 1))

        lrgt1 = self.fusion(lrgt_gap, msgt_gap1, msgt_gapx)
        lrgt1 = self.res1(lrgt1)

        lrgt1 = self.sede2(lrgt1)
        lrgt1, hrms = self.gmsf2(lrgt1, hrms)
        lrgt1, hrms = self.cssa1(lrgt1, hrms)

        lrgt1, hrms = self.gmsf3(lrgt1, hrms)
        lrgt1, hrms = self.cssa2(lrgt1, hrms)

        out = self.gt_out(lrgt1) + lrout

        return out

    def Grad_x(self, x):
        gra_x = torch.cat((x[:, :, 1:, :], x[:, :, -1, :].unsqueeze(2)), dim=2) - x

        return gra_x

    def Grad_y(self, x):
        gra_y = torch.cat((x[:, :, :, 1:], x[:, :, :, -1].unsqueeze(3)), dim=3) - x

        return gra_y

